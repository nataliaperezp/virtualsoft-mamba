import os
import json
import time
import argparse
import gcsfs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

import pyarrow.parquet as pq
import numpy as np
import polars as pl
import psutil
from sklearn.metrics import precision_recall_fscore_support, classification_report
import wandb

from model_arch import MambaModel

# ---------------------------------------------------------------------------
# DDP SETUP
# ---------------------------------------------------------------------------
def setup_ddp():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return True, local_rank, rank, world_size
    return False, 0, 0, 1

is_ddp, local_rank, rank, world_size = setup_ddp()
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# ARGUMENTOS
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--usar_balance", action="store_true", default=None)
parser.add_argument("--epochs",       type=int,   default=None)
parser.add_argument("--batch-size",   type=int,   default=None)
parser.add_argument("--num-workers",  type=int,   default=None)
parser.add_argument("--d-model",      type=int,   default=256)
parser.add_argument("--d-state",      type=int,   default=64)
parser.add_argument("--lr",           type=float, default=5e-4)
parser.add_argument("--grad-accum-steps", type=int, default=None, help="Pasos de acumulación de gradiente (effective_batch = batch_size * grad_accum)")
parser.add_argument("--max-seq-len",  type=int,   default=None, help="Tamaño máximo de ventana para evitar OOM")
parser.add_argument("--n-products",   type=int,   default=500000, help="Tamaño del vocabulario de productos fijos (evita escaneo)")
parser.add_argument("--product-csv-path", type=str, default=None, help="Ruta al CSV con la lista de todos los IDs validos de productos")
parser.add_argument("--test",         action="store_true", default=None, help="Modo prueba: usa solo 4 archivos parquet")
args, _ = parser.parse_known_args()

USAR_BALANCE = args.usar_balance if args.usar_balance is not None else os.environ.get("USAR_BALANCE", "False").lower() in ("true", "1", "t")
N_EPOCHS     = args.epochs       if args.epochs       is not None else int(os.environ.get("N_EPOCHS", "3"))
BATCH_SIZE   = args.batch_size   if args.batch_size   is not None else int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS  = args.num_workers  if args.num_workers  is not None else int(os.environ.get("NUM_WORKERS", "2"))
GRAD_ACCUM   = args.grad_accum_steps if args.grad_accum_steps is not None else int(os.environ.get("GRAD_ACCUM_STEPS", "8"))

# Con DDP, cada GPU procesa su propio micro-batch, así que el effective batch
# ya se multiplica por world_size. Ajustamos grad_accum para mantener el mismo
# batch efectivo total: eff_batch = BATCH_SIZE * world_size * GRAD_ACCUM
if is_ddp and world_size > 1:
    GRAD_ACCUM = max(1, GRAD_ACCUM // world_size)

MAX_SEQ_LEN  = args.max_seq_len if args.max_seq_len is not None else int(os.environ.get("MAX_SEQ_LEN", "2048"))
N_PRODUCTS   = args.n_products

OUT_DIR       = os.environ.get("OUT_DIR", "gs://ml-bucketvs/mamba-exp/processed")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mamba-train-4B")
GCS_CKPT_DIR  = os.environ.get("GCS_CKPT_DIR", OUT_DIR + "/checkpoints")
UIDS_PATH     = os.environ.get("UIDS_PATH", "split_estratificado_uids.json")

if rank == 0 and WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY, relogin=True)

# ---------------------------------------------------------------------------
# CARGA DE UIDs
# ---------------------------------------------------------------------------
with open(UIDS_PATH, "r") as f:
    uids_dict  = json.load(f)
    train_uids = set(uids_dict["train"])
    test_uids  = set(uids_dict["test"])

if rank == 0:
    print(f"✅ UIDs cargados | Train: {len(train_uids):,} | Test: {len(test_uids):,}")
    eff_batch = BATCH_SIZE * world_size * GRAD_ACCUM
    print(f"🚀 MODO 4B | MicroBatch: {BATCH_SIZE} x {world_size} GPUs x GradAccum: {GRAD_ACCUM} = EffBatch: {eff_batch} | SeqMax: {MAX_SEQ_LEN} | Workers: {NUM_WORKERS} | DDP: {is_ddp} (Rank {rank}/{world_size})")

# ---------------------------------------------------------------------------
# REMAPEO DE IDs DE PRODUCTO (Vía CSV)
# ---------------------------------------------------------------------------
PRODUCT_CSV_PATH = args.product_csv_path or os.environ.get("PRODUCT_CSV_PATH", "/app/postgres_apipythondb_producto.csv")

product_id_map = {}
if PRODUCT_CSV_PATH and os.path.exists(PRODUCT_CSV_PATH):
    if rank == 0: print(f"Cargando IDs de productos desde CSV: {PRODUCT_CSV_PATH}...")
    import csv
    _all_prod_ids = []
    with open(PRODUCT_CSV_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip().isdigit():
                _all_prod_ids.append(int(row[0].strip()))
    
    _all_prod_ids = sorted(set(_all_prod_ids) - {0})
    product_id_map = {orig: new for new, orig in enumerate(_all_prod_ids, start=1)}
    N_PRODUCTS_REMAPPED = len(product_id_map)
    N_PRODUCTS = N_PRODUCTS_REMAPPED
    
    if rank == 0:
        print(f"✅ Remapeo exacto restaurado: {len(_all_prod_ids)} productos únicos detectados en el CSV.")
else:
    if rank == 0:
        print("⚠️ No se proporcionó PRODUCT_CSV_PATH. Se usará hashing basado en --n-products.")
    N_PRODUCTS = args.n_products

# ---------------------------------------------------------------------------
# STREAMING DATASET CON BUCKET BATCHING
# ---------------------------------------------------------------------------
class StreamingBucketDataset(IterableDataset):
    def __init__(self, file_paths, uids_to_keep, max_seq_len, batch_size, use_balance, chunking=True):
        self.file_paths = file_paths
        self.uids_to_keep = uids_to_keep
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.bucket_size = batch_size * 3  # Reducido para evitar OOM en g2-standard-16 (64GB RAM)
        self.use_balance = use_balance
        self.chunking = chunking  # True=training (múltiples chunks), False=val (truncar al último)
        
        # Usar el mapa calculado globalmente
        self.product_map = product_id_map
        
    def _build_num_feats(self, nf_data):
        if not nf_data:
            return [[0.0, 0.0, 0.0] if self.use_balance else [0.0, 0.0]]
        res = []
        for d in nf_data:
            if isinstance(d, dict):
                row = [d.get("VALUE_NORM", 0.0), d.get("TIME_DELTA_NORM", 0.0)]
                if self.use_balance: row.append(d.get("BALANCE_NORM", 0.0))
            else:
                row = list(d)
            res.append(row)
        return res

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Iniciar GCSFS aquí (local a cada worker) para evitar el error "This class is not fork-safe"
        local_fs = gcsfs.GCSFileSystem()
        
        # Repartición de archivos cruzada entre DDP Nodes y Workers locales
        global_worker_id = rank * num_workers + worker_id
        total_workers = world_size * num_workers
        
        my_files = [f for i, f in enumerate(self.file_paths) if i % total_workers == global_worker_id]
        
        bucket = []
        for file_path in my_files:
            path_no_gs = file_path.replace("gs://", "")
            try:
                with local_fs.open(path_no_gs, 'rb') as f:
                    parquet_file = pq.ParquetFile(f)
                    import gc
                    # Leer en chunks pequeños para no saturar RAM
                    # No pedir TARGET_EVENT_SEQ porque los Parquets de inferencia pura no lo tienen
                    for batch in parquet_file.iter_batches(batch_size=1000, columns=["USER_ID", "INPUT_EVENT_SEQ", "INPUT_PRODUCT_SEQ", "NUM_FEATS_SEQ"]):
                        for row in batch.to_pylist():
                            full_e_seq = row["INPUT_EVENT_SEQ"]
                            
                            if full_e_seq is None or len(full_e_seq) < 2:
                                continue
                            
                            # Mamba es un modelo Causal: El target es el input desplazado 1 paso hacia el futuro
                            e_seq = full_e_seq[:-1]
                            t_seq = full_e_seq[1:]
                            
                            p_seq = row["INPUT_PRODUCT_SEQ"][:-1]
                            nf_seq = row["NUM_FEATS_SEQ"][:-1]
                            
                            # CHUNKING: Dividir en ventanas no-superpuestas de max_seq_len
                            # para que el modelo entrene con TODOS los eventos del usuario
                            seq_len = len(e_seq)
                            if not self.chunking or seq_len <= self.max_seq_len:
                                # Validación o secuencia corta: tomar los últimos max_seq_len
                                if seq_len > self.max_seq_len:
                                    e_seq = e_seq[-self.max_seq_len:]
                                    p_seq = p_seq[-self.max_seq_len:]
                                    nf_seq = nf_seq[-self.max_seq_len:]
                                    t_seq = t_seq[-self.max_seq_len:]
                                chunks = [(e_seq, p_seq, nf_seq, t_seq)]
                            else:
                                chunks = []
                                for start in range(0, seq_len, self.max_seq_len):
                                    end = min(start + self.max_seq_len, seq_len)
                                    if end - start < 2:  # chunk demasiado corto, saltar
                                        continue
                                    chunks.append((
                                        e_seq[start:end],
                                        p_seq[start:end],
                                        nf_seq[start:end],
                                        t_seq[start:end],
                                    ))

                            for c_e, c_p, c_nf, c_t in chunks:
                                # Mapeo de productos
                                if self.product_map:
                                    p_safe = [self.product_map.get(p, 0) for p in c_p]
                                else:
                                    p_safe = [(p % (N_PRODUCTS - 1)) + 1 if p > 0 else 0 for p in c_p]

                                nf_proc = self._build_num_feats(c_nf)

                                bucket.append({
                                    "length": len(c_e),
                                    "event_type": torch.tensor(c_e, dtype=torch.long),
                                    "product_id": torch.tensor(p_safe, dtype=torch.long),
                                    "num_feats": torch.tensor(nf_proc, dtype=torch.float32),
                                    "target": torch.tensor(c_t, dtype=torch.long)
                                })
                            
                            # Cuando el bucket se llena, ordena por longitud y genera batches (Smart Batching)
                            if len(bucket) >= self.bucket_size:
                                bucket.sort(key=lambda x: x["length"])
                                for i in range(0, len(bucket), self.batch_size):
                                    chunk = bucket[i:i + self.batch_size]
                                    if len(chunk) == self.batch_size:
                                        yield self._collate(chunk)
                                # Conservar remanente
                                bucket = bucket[-(len(bucket) % self.batch_size):]
                        
                        # Liberar memoria de PyArrow activamente antes del siguiente batch
                        del batch
                        gc.collect()
            except Exception as e:
                print(f"[Worker {global_worker_id}] Error en {file_path}: {e}")
                
        # Remanente final
        if bucket:
            bucket.sort(key=lambda x: x["length"])
            for i in range(0, len(bucket), self.batch_size):
                chunk = bucket[i:i + self.batch_size]
                if len(chunk) > 0:
                    yield self._collate(chunk)

    def _collate(self, batch):
        keys = ["event_type", "product_id", "num_feats", "target"]
        output = {}
        for key in keys:
            sequences = [item[key] for item in batch]
            output[key] = pad_sequence(sequences, batch_first=True, padding_value=0)
        return output

# ---------------------------------------------------------------------------
# FOCAL LOSS (Optimizada para Next Action)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.5, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index, weight=self.weight)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

# ---------------------------------------------------------------------------
# ENTRENAMIENTO PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    fs = gcsfs.GCSFileSystem()
    
    # Manejar el caso donde OUT_DIR ya incluya el comodín o termine en parquet
    clean_out_dir = OUT_DIR.replace("gs://", "")
    if clean_out_dir.endswith(".parquet"):
        gcs_glob = clean_out_dir
    else:
        gcs_glob = clean_out_dir.rstrip("/") + "/batch_*.parquet"
        
    all_files = sorted(["gs://" + p for p in fs.glob(gcs_glob)])
    
    TEST_MODE = args.test if args.test else os.environ.get("TEST_MODE", "False").lower() in ("true", "1", "t")
    
    if TEST_MODE and len(all_files) > 0:
        if rank == 0: print("⚠️ MODO PRUEBA ACTIVADO: Usando exactamente 1 solo archivo Parquet para la prueba más rápida posible.")
        train_files = [all_files[0]]
        val_files = [all_files[0]] # Reutilizamos el mismo archivo para que validation no explote
    else:
        split_idx = int(len(all_files) * 0.8)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
    
    if rank == 0:
        print(f"Archivos encontrados: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)}")
        # Guardar el split de archivos en GCS para reproducibilidad
        split_dict = {
            "train_files": train_files,
            "test_files": val_files
        }
        clean_out_dir = OUT_DIR.replace("gs://", "")
        base_dir = clean_out_dir if not clean_out_dir.endswith(".parquet") else clean_out_dir.rsplit("/", 1)[0]
        split_path = base_dir.rstrip("/") + "/dataset_splits.json"
        try:
            import json
            with fs.open(split_path, "w") as f:
                json.dump(split_dict, f, indent=4)
            print(f"✅ Split de datos guardado exitosamente en: gs://{split_path}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar el split en GCS: {e}")

    # Model Initialization
    model = MambaModel(
        n_products=N_PRODUCTS, n_event_types=21,
        d_model=args.d_model, d_state=args.d_state,
        use_balance=USAR_BALANCE
    ).to(device)
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    # Weights for classes
    counts = torch.ones(22)
    # Ejemplo pesos hardcodeados
    support_dict = {1: 77736, 2: 431737, 3: 219415, 4: 114894, 5: 27734, 17: 221349, 21: 15876}
    for ev_id, count in support_dict.items():
        if ev_id <= 21: counts[ev_id] = count
    weights = torch.log1p(counts.sum() / (21 * counts))
    weights[21] *= 2.5
    weights[0] = 0.0
    weights = weights.to(device)
    criterion = FocalLoss(weight=weights)

    if rank == 0 and WANDB_API_KEY:
        run = wandb.init(project=WANDB_PROJECT, config=vars(args))
        
    best_score = -1.0
    patience_counter = 0
    patience_limit = 3

    for epoch in range(N_EPOCHS):
        if rank == 0: print(f"\n► Epoch {epoch + 1}/{N_EPOCHS}")
        epoch_start = time.time()
        
        # --- TRAIN (con Gradient Accumulation) ---
        model.train()
        train_ds = StreamingBucketDataset(train_files, train_uids, MAX_SEQ_LEN, BATCH_SIZE, USAR_BALANCE)
        train_loader = DataLoader(train_ds, batch_size=None, num_workers=NUM_WORKERS, prefetch_factor=2)
        
        loss_accum = 0.0
        n_batches = 0
        n_optimizer_steps = 0
        optimizer.zero_grad()
        
        for batch in train_loader:
            et = batch["event_type"].to(device)
            pid = batch["product_id"].to(device)
            nf = batch["num_feats"].to(device)
            target = batch["target"].to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(et, pid, nf)
                min_seq = min(logits.shape[1], target.shape[1])
                loss = criterion(logits[:, :min_seq, :].reshape(-1, logits.size(-1)), target[:, :min_seq].reshape(-1))
                loss = loss / GRAD_ACCUM  # Normalizar la loss por los pasos de acumulación
                
            scaler.scale(loss).backward()
            
            loss_accum += loss.item() * GRAD_ACCUM  # Reportar la loss real (sin normalizar)
            n_batches += 1
            
            # Actualizar pesos solo cada GRAD_ACCUM micro-batches
            if n_batches % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                n_optimizer_steps += 1
            
            if rank == 0 and n_batches % (100 * GRAD_ACCUM) == 0:
                print(f"  Step {n_optimizer_steps} (micro-batch {n_batches}) | Loss: {loss_accum/n_batches:.4f}")
        
        # Flush gradientes residuales del último grupo incompleto
        if n_batches % GRAD_ACCUM != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
                
        # --- VALIDATION ---
        # Barrera para que ambas GPUs terminen training antes de empezar validación
        if is_ddp:
            dist.barrier()
        model.eval()
        val_ds = StreamingBucketDataset(val_files, test_uids, MAX_SEQ_LEN, BATCH_SIZE, USAR_BALANCE, chunking=True)
        val_loader = DataLoader(val_ds, batch_size=None, num_workers=min(NUM_WORKERS, 2), prefetch_factor=2)
        
        # Matriz de confusión incremental — memoria constante O(n_classes²)
        n_classes = 21  # número de event types
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        
        with torch.no_grad():
            for batch in val_loader:
                et = batch["event_type"].to(device)
                pid = batch["product_id"].to(device)
                nf = batch["num_feats"].to(device)
                target = batch["target"].to(device)
                
                with torch.cuda.amp.autocast():
                    logits = model(et, pid, nf)
                preds = torch.argmax(logits, dim=-1)
                
                min_v = min(preds.shape[1], target.shape[1])
                mask = (target[:, :min_v] != 0)
                t_flat = target[:, :min_v][mask].cpu().numpy()
                p_flat = preds[:, :min_v][mask].cpu().numpy()
                
                # Actualizar matriz de confusión (vectorizado, sin loop Python)
                valid = (t_flat < n_classes) & (p_flat < n_classes)
                np.add.at(confusion, (t_flat[valid], p_flat[valid]), 1)
                
                # Liberar tensores GPU inmediatamente
                del et, pid, nf, target, logits, preds
                
        # Aggregate across DDP
        if is_ddp:
            confusion_tensor = torch.tensor(confusion, device=device)
            dist.all_reduce(confusion_tensor, op=dist.ReduceOp.SUM)
            confusion = confusion_tensor.cpu().numpy()
        
        if rank == 0:
            # Calcular métricas desde la matriz de confusión
            per_class_support = confusion.sum(axis=1)
            per_class_tp = np.diag(confusion)
            per_class_pred = confusion.sum(axis=0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                per_class_precision = np.where(per_class_pred > 0, per_class_tp / per_class_pred, 0.0)
                per_class_recall = np.where(per_class_support > 0, per_class_tp / per_class_support, 0.0)
                per_class_f1 = np.where(
                    (per_class_precision + per_class_recall) > 0,
                    2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall),
                    0.0
                )
            
            # Macro: promedio sobre clases con soporte > 0
            active_classes = per_class_support > 0
            p_macro = per_class_precision[active_classes].mean() if active_classes.any() else 0.0
            r_macro = per_class_recall[active_classes].mean() if active_classes.any() else 0.0
            f1_m = per_class_f1[active_classes].mean() if active_classes.any() else 0.0
            
            # Weighted: promedio ponderado por soporte
            total_support = per_class_support.sum()
            if total_support > 0:
                p_weighted = (per_class_precision * per_class_support).sum() / total_support
                r_weighted = (per_class_recall * per_class_support).sum() / total_support
                f1_weighted = (per_class_f1 * per_class_support).sum() / total_support
            else:
                p_weighted = r_weighted = f1_weighted = 0.0
            
            epoch_duration = time.time() - epoch_start
            ram_usage = psutil.Process().memory_info().rss / (1024 ** 3)
            
            print(f"Epoch {epoch+1} | F1: {f1_m:.4f} | Recall: {r_macro:.4f} | {epoch_duration:.1f}s | RAM: {ram_usage:.1f}GB")
            
            if WANDB_API_KEY:
                class_rows = [
                    [str(i), float(per_class_precision[i]), float(per_class_recall[i]), float(per_class_f1[i]), int(per_class_support[i])]
                    for i in range(n_classes) if per_class_support[i] > 0
                ]
                wandb.log({
                    "epoch": epoch+1, 
                    "train_loss": loss_accum/max(n_batches, 1), 
                    "val_f1_macro": f1_m, 
                    "val_recall_macro": r_macro,
                    "val_precision_macro": p_macro,
                    "val_f1_weighted": f1_weighted,
                    "val_recall_weighted": r_weighted,
                    "val_precision_weighted": p_weighted,
                    "ram_gb": ram_usage,
                    "epoch_duration_sec": epoch_duration,
                    "per_class_metrics": wandb.Table(columns=["class", "precision", "recall", "f1", "support"], data=class_rows)
                })
                
            if r_macro > best_score:
                best_score = r_macro
                ckpt_path = "/tmp/best_model_4B.ckpt"
                
                # Unwrap DDP model for saving
                model_to_save = model.module if is_ddp else model
                
                # Incluir el código fuente de las clases para inferencia (Portabilidad Vertex AI)
                _arch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_arch.py")
                try:
                    with open(_arch_path, "r") as _f:
                        _model_arch_source = _f.read()
                except Exception:
                    _model_arch_source = ""
                
                torch.save({
                    "state_dict": model_to_save.state_dict(),
                    "config_arch": {"n_products": N_PRODUCTS, "n_event_types": 21, "d_model": args.d_model, "d_state": args.d_state, "use_balance": USAR_BALANCE},
                    "config": vars(args),
                    "product_id_map": product_id_map,
                    "model_arch_source": _model_arch_source,
                    "metric": r_macro,
                    "metric_name": "recall_macro",
                    "recall_macro": r_macro,
                    "recall_weighted": r_weighted,
                    "precision_macro": p_macro,
                    "precision_weighted": p_weighted,
                    "f1_macro": f1_m,
                    "f1_weighted": f1_weighted,
                    "experiment": {
                        "exp": 1,
                        "task": "next_action",
                        "usar_balance": USAR_BALANCE,
                        "run": run.name if WANDB_API_KEY else "4B_run",
                        "epoch": epoch + 1,
                    }
                }, ckpt_path)
                
                gcs_path = GCS_CKPT_DIR.replace("gs://", "") + "/best_model_4B.ckpt"
                fs.put(ckpt_path, gcs_path)
                print(f"🏆 Modelo guardado en gs://{gcs_path}")
            else:
                patience_counter += 1
                print(f"⚠️ Early stopping patience: {patience_counter}/{patience_limit}")

        # --- EARLY STOPPING SINCRONIZADO PARA DDP ---
        # Barrera para que ambas GPUs terminen validación antes de sincronizar
        if is_ddp:
            dist.barrier()
        stop_signal = torch.tensor([0], device=device)
        if rank == 0 and patience_counter >= patience_limit:
            print("🛑 Early stopping activado! Deteniendo el entrenamiento en todas las GPUs.")
            stop_signal[0] = 1
            
        if is_ddp:
            dist.broadcast(stop_signal, src=0)
            
        if stop_signal.item() == 1:
            break

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
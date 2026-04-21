import os
import json
import gc
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl
import psutil
import optuna
import gcsfs


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from mamba_ssm import Mamba
from sklearn.metrics import precision_recall_fscore_support, classification_report
import wandb

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE ENTORNO
# En Vertex AI las variables se inyectan en el Job spec.
# Localmente puedes usar un .env y exportarlas antes de correr.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# ARGUMENTOS DE EXPERIMENTO
# Prioridad: CLI args > env vars > defaults
# Uso local: python Mamba.py --exp 2 --task both --usar_balance
# En Vertex AI: se inyectan como env vars en el job spec (sin necesidad de args)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="Mamba Training Job")
_parser.add_argument("--exp",          type=int,   default=None, help="Número de experimento")
_parser.add_argument("--task",         type=str,   default=None, choices=["next_action", "contrastive", "both"])
_parser.add_argument("--usar_balance", action="store_true", default=None, help="Incluir BALANCE_NORM como feature")
_parser.add_argument("--test",         action="store_true", default=None,
                     help="Modo prueba: pocos archivos, 2 trials, 2 epochs")
_parser.add_argument("--test-files",   type=int,   default=None,
                     help="Número de archivos parquet a usar en modo prueba (default: 10 ≈ 1000 usuarios)")
_parser.add_argument("--group-size",   type=int,   default=None,
                     help="Archivos parquet por grupo de carga (default: 2 en prueba, 100 en prod)")
_parser.add_argument("--epochs",       type=int,   default=None,
                     help="Épocas de entrenamiento (default: 2 en prueba, 10 en prod)")
_parser.add_argument("--batch-size",   type=int,   default=None,
                     help="Batch size (default: 64 en prueba, 512 en prod)")
_parser.add_argument("--num-workers",  type=int,   default=None,
                     help="Workers del DataLoader (default: 6)")
_parser.add_argument("--trials",       type=int,   default=None,
                     help="Número de trials de Optuna (default: 2 en prueba, 15 en prod)")
_parser.add_argument("--use-sweep",    action="store_true", default=None,
                     help="Usar W&B Sweep (Bayesiano) en lugar de Optuna")
# Modo estático (sin Optuna)
_parser.add_argument("--static",       action="store_true", default=None,
                     help="Modo estático: entrena con hiperparámetros fijos, sin Optuna")
_parser.add_argument("--d-model",      type=int,   default=None, help="d_model fijo (modo estático)")
_parser.add_argument("--d-state",      type=int,   default=None, help="d_state fijo (modo estático)")
_parser.add_argument("--lr",           type=float, default=None, help="learning rate fijo (modo estático)")
_parser.add_argument("--lambda-cl",    type=float, default=None, help="lambda contrastive fijo (modo estático)")
_parser.add_argument("--temp",         type=float, default=None, help="temperatura contrastive fija (modo estático)")
_args, _ = _parser.parse_known_args()

EXP          = _args.exp        if _args.exp        is not None else int(os.environ.get("EXP",  "1"))
task         = _args.task       if _args.task       is not None else os.environ.get("TASK", "next_action").lower()
USAR_BALANCE = _args.usar_balance if _args.usar_balance         else os.environ.get("USAR_BALANCE", "False").lower() in ("true", "1", "t")
TEST_MODE    = _args.test       if _args.test       is not None else os.environ.get("TEST_MODE", "False").lower() in ("true", "1", "t")
TEST_FILES   = _args.test_files if _args.test_files is not None else int(os.environ.get("TEST_FILES", "10"))
GROUP_SIZE   = _args.group_size if _args.group_size is not None else int(os.environ.get("GROUP_SIZE", "0"))
N_EPOCHS     = _args.epochs     if _args.epochs     is not None else int(os.environ.get("N_EPOCHS", "0"))
BATCH_SIZE   = _args.batch_size if _args.batch_size is not None else int(os.environ.get("BATCH_SIZE", "0"))
NUM_WORKERS  = _args.num_workers if _args.num_workers is not None else int(os.environ.get("NUM_WORKERS", "6"))
N_TRIALS     = _args.trials     if _args.trials     is not None else int(os.environ.get("N_TRIALS", "0"))
USE_SWEEP    = _args.use_sweep  if _args.use_sweep  is not None else os.environ.get("USE_SWEEP", "False").lower() in ("true", "1", "t")
# Modo estático
STATIC_MODE  = _args.static     if _args.static     is not None else os.environ.get("STATIC_MODE", "False").lower() in ("true", "1", "t")
STATIC_D_MODEL    = _args.d_model    if _args.d_model    is not None else int(os.environ.get("STATIC_D_MODEL",    "256"))
STATIC_D_STATE    = _args.d_state    if _args.d_state    is not None else int(os.environ.get("STATIC_D_STATE",    "64"))
STATIC_LR         = _args.lr         if _args.lr         is not None else float(os.environ.get("STATIC_LR",       "5e-4"))
STATIC_LAMBDA_CL  = _args.lambda_cl  if _args.lambda_cl  is not None else float(os.environ.get("STATIC_LAMBDA_CL","0.5"))
STATIC_TEMP       = _args.temp       if _args.temp       is not None else float(os.environ.get("STATIC_TEMP",     "0.1"))
# GROUP_SIZE=0, N_EPOCHS=0, BATCH_SIZE=0, N_TRIALS=0 → default según modo

OUT_DIR       = os.environ["OUT_DIR"]            # gs://bucket/path/sequences_output
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mamba-train-opt")
GCS_CKPT_DIR  = os.environ.get("GCS_CKPT_DIR", OUT_DIR + "/checkpoints")
UIDS_PATH     = os.environ.get("UIDS_PATH", "split_estratificado_uids.json")

os.environ["POLARS_MAX_THREADS"] = "1"

# ---------------------------------------------------------------------------
# W&B LOGIN (sin prompts interactivos — requerido en Vertex)
# ---------------------------------------------------------------------------
wandb.login(key=WANDB_API_KEY, relogin=True)

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# CARGA LAZY DEL DATASET
# ---------------------------------------------------------------------------
events_final = pl.scan_parquet(f"{OUT_DIR}/batch_*.parquet")
print(f"✅ Dataset Lazy cargado desde {OUT_DIR}")

# ---------------------------------------------------------------------------
# REMAPEO DE IDs DE PRODUCTO
# El ID original llega hasta ~440k pero hay <10k productos únicos.
# Construimos un mapa {id_original → id_compacto (1..N_unique)} para reducir
# la tabla de embeddings de ~440k filas a solo N_unique filas.
# El mapa se guarda en el checkpoint para aplicarlo en inferencia.
# ---------------------------------------------------------------------------
_all_prod_ids = sorted(set(
    events_final
    .select(pl.col("INPUT_PRODUCT_SEQ").explode())
    .unique()
    .collect()
    ["INPUT_PRODUCT_SEQ"]
    .drop_nulls()
    .to_list()
) - {0})   # 0 es padding, lo excluimos
product_id_map: dict[int, int] = {orig: new for new, orig in enumerate(_all_prod_ids, start=1)}
N_PRODUCTS_REMAPPED = len(product_id_map)
print(f"✅ Remapeo de productos: {440874} IDs posibles → {N_PRODUCTS_REMAPPED} únicos reales")

# ---------------------------------------------------------------------------
# CLASES
# ---------------------------------------------------------------------------

from model_arch import EventTokenizer, MambaModel  # noqa: E402


class FastSequenceDataset(Dataset):
    def __init__(self, df, task="next_action"):
        self.task = task
        self.event_seqs  = df["INPUT_EVENT_SEQ"].to_list()
        self.prod_seqs   = [
            [product_id_map.get(p, 0) for p in seq]
            for seq in df["INPUT_PRODUCT_SEQ"].to_list()
        ]
        self.target_seqs = df["TARGET_EVENT_SEQ"].to_list()

        self.num_feats_seqs = []
        for seq in df["NUM_FEATS_SEQ"].to_list():
            tensor_feat = torch.tensor([
                [f["VALUE_NORM"], f["TIME_DELTA_NORM"], f["BALANCE_NORM"]]
                for f in seq
            ], dtype=torch.float32)
            self.num_feats_seqs.append(tensor_feat)

    def __len__(self):
        return len(self.event_seqs)

    def __getitem__(self, idx):
        event_x    = torch.tensor(self.event_seqs[idx],  dtype=torch.long)
        prod_x     = torch.tensor(self.prod_seqs[idx],   dtype=torch.long)
        target     = torch.tensor(self.target_seqs[idx], dtype=torch.long)
        num_feats_x = self.num_feats_seqs[idx]

        batch = {
            "event_type": event_x,
            "product_id": prod_x,
            "num_feats":  num_feats_x,
            "target":     target
        }

        if self.task in ["contrastive", "both"]:
            batch["event_type_v2"] = event_x.clone()
            batch["product_id_v2"] = prod_x.clone()
            batch["num_feats_v2"]  = num_feats_x.clone()

        return batch


def collate_fn(batch):
    keys   = batch[0].keys()
    output = {}
    for key in keys:
        sequences    = [item[key] for item in batch]
        padded       = pad_sequence(sequences, batch_first=True, padding_value=0)
        output[key]  = padded
    return output


def get_dataloader_for_multiple_files(file_list, uids_to_keep, batch_size=512, task="both", num_workers=6):
    lazy_dfs = [pl.scan_parquet(f) for f in file_list]
    df_combined = pl.concat(lazy_dfs).filter(
        pl.col("USER_ID").is_in(list(uids_to_keep))
    ).collect()

    if len(df_combined) == 0:
        return None

    ds = FastSequenceDataset(df_combined, task=task)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        prefetch_factor=1 if num_workers > 0 else None,
        persistent_workers=False
    )


# ---------------------------------------------------------------------------
# CARGA DE UIDs (puede ser ruta local dentro del contenedor o GCS)
# ---------------------------------------------------------------------------
with open(UIDS_PATH, "r") as f:
    uids_dict  = json.load(f)
    train_uids = set(uids_dict["train"])
    test_uids  = set(uids_dict["test"])

sequences_lf  = pl.scan_parquet(f"{OUT_DIR}/batch_*.parquet")
train_count = sequences_lf.filter(pl.col("USER_ID").is_in(list(train_uids))).select(pl.len()).collect().item()
test_count  = sequences_lf.filter(pl.col("USER_ID").is_in(list(test_uids))).select(pl.len()).collect().item()
print(f"✅ Train: {train_count:,} filas | Test: {test_count:,} filas")

# ---------------------------------------------------------------------------
# CONSTANTES GLOBALES
# ---------------------------------------------------------------------------
N_PRODUCTS    = N_PRODUCTS_REMAPPED   # cardinalidad real de productos únicos (remapeados)
n_event_types = 21       # máximo ID de evento (usado para pesos de clases)

print(f"🧪 Experimento {EXP} | task={task} | usar_balance={USAR_BALANCE}")

support_dict = {
    1: 77736,  2: 431737, 3: 219415, 4: 114894,  5: 27734,
    6: 116,    7: 11511,  8: 370,    9: 16691,   11: 16223,
    13: 159,   14: 1,     15: 625,   17: 221349,  18: 2652,
    19: 2459,  20: 6774,  21: 15876
}

# ---------------------------------------------------------------------------
# PESOS DE CLASES
# ---------------------------------------------------------------------------
counts = torch.ones(n_event_types + 1)
for ev_id, count in support_dict.items():
    if ev_id <= n_event_types:   # <= para incluir la clase 21
        counts[ev_id] = count

total_samples = counts.sum()
weights = torch.log1p(total_samples / (n_event_types * counts))
weights[21] = weights[21] * 2.5   # refuerzo extra para clase de riesgo
weights[0]  = 0.0                  # padding no penaliza
weights = weights.to(device)


# ---------------------------------------------------------------------------
# FOCAL LOSS
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.5, ignore_index=0):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(
            inputs, targets,
            reduction='none',
            ignore_index=self.ignore_index,
            weight=self.weight
        )
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


criterion = FocalLoss(weight=weights, gamma=2.5, ignore_index=0)

# ---------------------------------------------------------------------------
# SPLIT DE ARCHIVOS (GCS)
# ---------------------------------------------------------------------------
fs        = gcsfs.GCSFileSystem()
gcs_glob  = OUT_DIR.replace("gs://", "") + "/batch_*.parquet"
all_files = sorted(["gs://" + p for p in fs.glob(gcs_glob)])

if len(all_files) < 2:
    raise ValueError("Se necesitan al menos 2 archivos Parquet para tener train y val.")

print(f"Archivos encontrados: {len(all_files)}")

if TEST_MODE:
    all_files = all_files[:TEST_FILES]
    default_group_size = 2
    default_batch_size = 64
    print(f"⚠️  MODO PRUEBA: usando {len(all_files)} archivos (~{len(all_files) * 100} usuarios)")
else:
    default_group_size = 100
    default_batch_size = 512

group_size = GROUP_SIZE if GROUP_SIZE > 0 else default_group_size
batch_size = BATCH_SIZE if BATCH_SIZE > 0 else default_batch_size

split_idx   = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files   = all_files[split_idx:]

if len(val_files) == 0:
    val_files = train_files[-1:]
    train_files = train_files[:-1]

# Agrupamos en chunks para no saturar RAM
train_groups = [train_files[i:i + group_size] for i in range(0, len(train_files), group_size)]
val_groups   = [val_files[i:i + group_size]   for i in range(0, len(val_files),   group_size)]

mode_label = "PRUEBA" if TEST_MODE else "PRODUCCIÓN"
print(f"🚀 MODO {mode_label}: {len(train_files)} train | {len(val_files)} val | grupos de {group_size}")


# ---------------------------------------------------------------------------
# EARLY STOPPING
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter    = 0


# ---------------------------------------------------------------------------
# FUNCIONES DE APOYO
# ---------------------------------------------------------------------------
def apply_event_masking(event_seq, mask_prob=0.15):
    masked_seq = event_seq.clone()
    mask = torch.rand(masked_seq.shape).to(event_seq.device) < mask_prob
    masked_seq[mask] = 0
    return masked_seq


def get_pooled_embedding(emb):
    return emb.mean(dim=1) if emb.dim() == 3 else emb


def compute_contrastive_loss(emb1, emb2, temp=0.07):
    emb1, emb2 = F.normalize(emb1, dim=1), F.normalize(emb2, dim=1)
    logits      = torch.matmul(emb1, emb2.T) / temp
    labels      = torch.arange(emb1.size(0)).to(emb1.device)
    return F.cross_entropy(logits, labels)


def upload_checkpoint_to_gcs(local_path: str, gcs_filename: str):
    """Sube un checkpoint local a GCS y elimina el archivo temporal."""
    balance_tag = "balance" if USAR_BALANCE else "nobalance"
    gcs_path = GCS_CKPT_DIR.replace("gs://", "") + f"/exp{EXP}_{task}_{balance_tag}/{gcs_filename}"
    fs.put(local_path, gcs_path)
    os.remove(local_path)
    print(f"✅ Checkpoint subido a gs://{gcs_path}")


# ---------------------------------------------------------------------------
# MEJOR MODELO GLOBAL (entre todos los runs/trials)
# ---------------------------------------------------------------------------
best_overall_recall = -1.0

# ---------------------------------------------------------------------------
# NÚCLEO DE ENTRENAMIENTO (compartido por Optuna, W&B Sweep y modo estático)
# on_epoch_end(metric, epoch) -> bool  — devuelve True para detener temprano
# ---------------------------------------------------------------------------
def _run_training(config, run, on_epoch_end=None):
    global best_overall_recall

    model     = MambaModel(
        n_products=N_PRODUCTS, n_event_types=21,
        d_model=config["d_model"], d_state=config["d_state"],
        use_balance=USAR_BALANCE
    ).to(device)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scaler        = torch.cuda.amp.GradScaler()
    early_stopper = EarlyStopping(patience=5)
    best_score    = -1.0

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        print(f"\n  ► Epoch {epoch + 1}/{config['epochs']} | {len(train_groups)} grupos de entrenamiento...")

        # ── ENTRENAMIENTO ──────────────────────────────────────────────────
        model.train()
        train_loss_accum = 0.0
        grad_norm_accum  = 0.0
        n_batches        = 0

        for g_idx, group in enumerate(train_groups):
            torch.cuda.empty_cache()
            loader = get_dataloader_for_multiple_files(
                group, train_uids, batch_size=config["batch_size"], task=task, num_workers=NUM_WORKERS
            )
            if loader is None:
                continue

            for batch in loader:
                optimizer.zero_grad()
                et     = batch["event_type"].to(device)
                pid    = batch["product_id"].to(device)
                nf     = batch["num_feats"].to(device)
                target = batch["target"].to(device)

                with torch.cuda.amp.autocast():
                    logits, emb1_raw = model(et, pid, nf, return_embeddings=True)
                    min_seq    = min(logits.shape[1], target.shape[1])
                    loss_total = 0.0

                    if task in ["next_action", "both"]:
                        loss_total += criterion(
                            logits[:, :min_seq, :].reshape(-1, logits.size(-1)),
                            target[:, :min_seq].reshape(-1)
                        )
                    if task in ["contrastive", "both"]:
                        et_v2       = apply_event_masking(et, mask_prob=0.4)
                        _, emb2_raw = model(et_v2, pid, nf, return_embeddings=True)
                        emb1, emb2 = emb1_raw, emb2_raw
                        loss_cl     = compute_contrastive_loss(emb1, emb2, temp=config["temp"])
                        loss_total += (config["lambda_cl"] * loss_cl) if task == "both" else loss_cl

                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                grad_norm        = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss_accum += loss_total.item()
                grad_norm_accum  += grad_norm.item()
                n_batches        += 1

            del loader
            gc.collect()

            ckpt_steps = max(1, len(train_groups) // 4)
            if (g_idx + 1) % ckpt_steps == 0 or (g_idx + 1) == len(train_groups):
                print(f"    Grupo {g_idx + 1}/{len(train_groups)} | Loss: {train_loss_accum / max(n_batches, 1):.4f}")

        avg_train_loss = train_loss_accum / max(n_batches, 1)
        avg_grad_norm  = grad_norm_accum  / max(n_batches, 1)

        # ── VALIDACIÓN ─────────────────────────────────────────────────────
        model.eval()
        ram_usage      = psutil.Process().memory_info().rss / (1024 ** 3)
        gpu_usage      = torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        epoch_duration = time.time() - epoch_start

        if task == "contrastive":
            val_cl_loss_accum = 0.0
            n_val_batches     = 0
            all_cosine_sims   = []

            with torch.no_grad():
                for v_group in val_groups:
                    loader_val = get_dataloader_for_multiple_files(
                        v_group, test_uids, batch_size=config["batch_size"], task=task, num_workers=NUM_WORKERS
                    )
                    if loader_val is None:
                        continue
                    for batch in loader_val:
                        et  = batch["event_type"].to(device)
                        pid = batch["product_id"].to(device)
                        nf  = batch["num_feats"].to(device)

                        with torch.cuda.amp.autocast():
                            emb1    = model.get_user_embedding(et, pid, nf)
                            et_v2   = apply_event_masking(et, mask_prob=0.15)
                            emb2    = model.get_user_embedding(et_v2, pid, nf)
                            cl_loss = compute_contrastive_loss(emb1, emb2, temp=config["temp"])

                        val_cl_loss_accum += cl_loss.item()
                        n_val_batches     += 1
                        # similitud coseno de pares positivos (embeddings ya normalizados)
                        all_cosine_sims.extend((emb1 * emb2).sum(dim=-1).cpu().tolist())
                    del loader_val

            val_cl_loss   = val_cl_loss_accum / max(n_val_batches, 1)
            current_score = -val_cl_loss
            f1_m = r_macro = p_macro = f1_weighted = r_weighted = p_weighted = 0.0

            print(
                f"{run.name} | Epoch {epoch + 1}/{config['epochs']} | "
                f"Val CL Loss: {val_cl_loss:.4f} | "
                f"RAM: {ram_usage:.2f} GB | GPU: {gpu_usage:.2f} GB | {epoch_duration:.1f}s"
            )

            log_dict = {
                "epoch":                    epoch + 1,
                "train_loss":               avg_train_loss,
                "grad_norm":                avg_grad_norm,
                "val_cl_loss":              val_cl_loss,
                "pos_pair_cosine_sim_mean": float(np.mean(all_cosine_sims)) if all_cosine_sims else 0.0,
                "ram_gb":                   ram_usage,
                "gpu_gb":                   gpu_usage,
                "epoch_duration_sec":       epoch_duration,
            }
            if all_cosine_sims:
                log_dict["pos_pair_cosine_sim"] = wandb.Histogram(all_cosine_sims)

        else:
            all_preds, all_targets = [], []

            with torch.no_grad():
                for v_group in val_groups:
                    loader_val = get_dataloader_for_multiple_files(
                        v_group, test_uids, batch_size=config["batch_size"], task=task, num_workers=NUM_WORKERS
                    )
                    if loader_val is None:
                        continue
                    for batch in loader_val:
                        et     = batch["event_type"].to(device)
                        pid    = batch["product_id"].to(device)
                        nf     = batch["num_feats"].to(device)
                        target = batch["target"].to(device)

                        with torch.cuda.amp.autocast():
                            logits = model(et, pid, nf)
                        preds  = torch.argmax(logits, dim=-1)
                        min_v  = min(preds.shape[1], target.shape[1])
                        mask   = (target[:, :min_v] != 0)
                        all_targets.append(target[:, :min_v][mask].cpu())
                        all_preds.append(preds[:, :min_v][mask].cpu())
                    del loader_val

            y_true = torch.cat(all_targets).numpy()
            y_pred = torch.cat(all_preds).numpy()

            p_macro,    r_macro,    f1_m,        _ = precision_recall_fscore_support(y_true, y_pred, average='macro',    zero_division=0)
            p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            current_score = r_macro

            # Tabla por clase
            report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            class_rows  = [
                [lbl, v["precision"], v["recall"], v["f1-score"], int(v["support"])]
                for lbl, v in report_dict.items()
                if lbl not in ("accuracy", "macro avg", "weighted avg")
            ]
            print(
                f"{run.name} | Epoch {epoch + 1}/{config['epochs']} | "
                f"F1 Macro: {f1_m:.4f} | Recall Macro: {r_macro:.4f} | Precision Macro: {p_macro:.4f} | "
                f"RAM: {ram_usage:.2f} GB | GPU: {gpu_usage:.2f} GB | {epoch_duration:.1f}s"
            )
            print(classification_report(y_true, y_pred, zero_division=0))

            log_dict = {
                "epoch":                  epoch + 1,
                "train_loss":             avg_train_loss,
                "grad_norm":              avg_grad_norm,
                "val_recall_macro":       r_macro,
                "val_recall_weighted":    r_weighted,
                "val_precision_macro":    p_macro,
                "val_precision_weighted": p_weighted,
                "val_f1_macro":           f1_m,
                "val_f1_weighted":        f1_weighted,
                "per_class_metrics":      wandb.Table(
                    columns=["class", "precision", "recall", "f1", "support"],
                    data=class_rows
                ),
                "ram_gb":                 ram_usage,
                "gpu_gb":                 gpu_usage,
                "epoch_duration_sec":     epoch_duration,
            }

        wandb.log(log_dict)

        # Checkpoint si mejoró
        if current_score > best_score:
            best_score = current_score

            # Incluir el código fuente de las clases para inferencia sin model_arch.py
            _arch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_arch.py")
            with open(_arch_path, "r") as _f:
                _model_arch_source = _f.read()

            checkpoint_payload = {
                "state_dict": model.state_dict(),        # ← pesos
                "config_arch": {                          # ← parámetros del constructor
                    "n_products":    N_PRODUCTS,
                    "n_event_types": 21,
                    "d_model":       config["d_model"],
                    "d_state":       config["d_state"],
                    "use_balance":   USAR_BALANCE,
                },
                "config":            config,    
                "product_id_map":    product_id_map,
                "model_arch_source": _model_arch_source,
                "metric":             current_score,
                "metric_name":        "neg_val_cl_loss" if task == "contrastive" else "recall_macro",
                "recall_macro":       r_macro,
                "recall_weighted":    r_weighted,
                "precision_macro":    p_macro,
                "precision_weighted": p_weighted,
                "f1_macro":           f1_m,
                "f1_weighted":        f1_weighted,
                "experiment": {
                    "exp":          EXP,
                    "task":         task,
                    "usar_balance": USAR_BALANCE,
                    "run":          run.name,
                    "epoch":        epoch + 1,
                }
            }
            local_ckpt = f"/tmp/best_model_{run.name}.ckpt"
            torch.save(checkpoint_payload, local_ckpt)
            upload_checkpoint_to_gcs(local_ckpt, f"best_model_{run.name}.ckpt")

            if current_score > best_overall_recall:
                best_overall_recall = current_score
                local_overall = "/tmp/best_model_overall.ckpt"
                torch.save(checkpoint_payload, local_overall)
                upload_checkpoint_to_gcs(local_overall, "best_model_overall.ckpt")
                print(f"🏆 Nuevo mejor modelo global: score={current_score:.4f} (run {run.name}, epoch {epoch + 1})")

        report_metric = current_score

        # Callback por epoch (usado por Optuna para pruning)
        if on_epoch_end is not None and on_epoch_end(report_metric, epoch):
            print(f"  Pruning signal recibido en epoch {epoch + 1}")
            break

        early_stopper(report_metric)
        if early_stopper.early_stop:
            print(f"Early stopping en epoch {epoch + 1}")
            break

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return best_score


# ---------------------------------------------------------------------------
# WRAPPER OPTUNA
# ---------------------------------------------------------------------------
def objective(trial):
    config = {
        "d_model":    trial.suggest_categorical("d_model", [128, 256, 512]) if not TEST_MODE else 128,
        "d_state":    trial.suggest_int("d_state", 16, 64, step=16)         if not TEST_MODE else 16,
        "lambda_cl":  trial.suggest_float("lambda_cl", 0.1, 0.8)            if not TEST_MODE else 0.5,
        "lr":         trial.suggest_float("lr", 1e-4, 1e-3, log=True)       if not TEST_MODE else 1e-3,
        "temp":       trial.suggest_float("temp", 0.05, 0.15)               if not TEST_MODE else 0.07,
        "batch_size": batch_size,
        "epochs":     N_EPOCHS if N_EPOCHS > 0 else (2 if TEST_MODE else 10),
    }

    run = wandb.init(
        project=f"{WANDB_PROJECT}-{EXP}-{task}-{USAR_BALANCE}",
        name=f"trial-{trial.number}",
        config=config,
        reinit=True
    )

    print(
        f"\n{'='*60}\n"
        f"  Trial {trial.number} | d_model={config['d_model']} | d_state={config['d_state']} "
        f"| lr={config['lr']:.2e} | lambda_cl={config['lambda_cl']:.2f} | temp={config['temp']:.2f}"
        f"\n{'='*60}"
    )

    pruned = [False]

    def on_epoch_end(metric, epoch):
        trial.report(metric, epoch)
        if trial.should_prune():
            pruned[0] = True
            return True
        return False

    best_score = _run_training(config, run, on_epoch_end=on_epoch_end)
    run.finish()

    if pruned[0]:
        raise optuna.exceptions.TrialPruned()

    return best_score


# ---------------------------------------------------------------------------
# WRAPPER W&B SWEEP
# ---------------------------------------------------------------------------
def train_one_trial():
    run = wandb.init()
    wc  = wandb.config

    n_epochs = N_EPOCHS if N_EPOCHS > 0 else (2 if TEST_MODE else 10)
    config = {
        "d_model":    wc.d_model,
        "d_state":    wc.d_state,
        "lambda_cl":  wc.lambda_cl,
        "lr":         wc.lr,
        "temp":       wc.temp,
        "batch_size": batch_size,
        "epochs":     n_epochs,
    }
    wandb.config.update({"batch_size": batch_size, "epochs": n_epochs}, allow_val_change=True)

    print(
        f"\n{'='*60}\n"
        f"  Sweep {run.name} | d_model={config['d_model']} | d_state={config['d_state']} "
        f"| lr={config['lr']:.2e} | lambda_cl={config['lambda_cl']:.2f} | temp={config['temp']:.2f}"
        f"\n{'='*60}"
    )

    _run_training(config, run)
    run.finish()


# ---------------------------------------------------------------------------
# ENTRENAMIENTO ESTÁTICO (sin optimización de hiperparámetros)
# ---------------------------------------------------------------------------
def train_static():
    n_epochs = N_EPOCHS if N_EPOCHS > 0 else (2 if TEST_MODE else 10)
    config = {
        "d_model":    STATIC_D_MODEL,
        "d_state":    STATIC_D_STATE,
        "lr":         STATIC_LR,
        "lambda_cl":  STATIC_LAMBDA_CL,
        "temp":       STATIC_TEMP,
        "batch_size": batch_size,
        "epochs":     n_epochs,
    }
    print(
        f"\n{'='*60}\n"
        f"  MODO ESTÁTICO | d_model={config['d_model']} | d_state={config['d_state']} "
        f"| lr={config['lr']:.2e} | lambda_cl={config['lambda_cl']:.2f} | temp={config['temp']:.2f}"
        f"\n{'='*60}"
    )
    run = wandb.init(
        project=f"{WANDB_PROJECT}-{EXP}-{task}-{USAR_BALANCE}",
        name="static-run",
        config=config,
        reinit=True
    )
    best_score = _run_training(config, run)
    run.finish()
    print(f"\n✅ Entrenamiento estático finalizado | Mejor score: {best_score:.4f}")
    return best_score


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if STATIC_MODE:
        print(f"⚙️  MODO ESTÁTICO | d_model={STATIC_D_MODEL} | d_state={STATIC_D_STATE} | lr={STATIC_LR:.2e} | lambda_cl={STATIC_LAMBDA_CL:.2f} | temp={STATIC_TEMP:.2f}")
        train_static()

    elif USE_SWEEP:
        n_trials = N_TRIALS if N_TRIALS > 0 else (2 if TEST_MODE else 15)

        if TEST_MODE:
            sweep_parameters = {
                "d_model":   {"value": 128},
                "d_state":   {"value": 16},
                "lambda_cl": {"value": 0.5},
                "lr":        {"value": 1e-3},
                "temp":      {"value": 0.07},
            }
        else:
            sweep_parameters = {
                "d_model":   {"values": [128, 256, 512]},
                "d_state":   {"values": [16, 32, 48, 64]},
                "lambda_cl": {"distribution": "uniform",            "min": 0.1,  "max": 0.8},
                "lr":        {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
                "temp":      {"distribution": "uniform",            "min": 0.05, "max": 0.15},
            }

        sweep_config = {
            "method": "bayes",
            "metric": {
                "name": "val_cl_loss"      if task == "contrastive" else "val_recall_macro",
                "goal": "minimize"         if task == "contrastive" else "maximize",
            },
            "parameters": sweep_parameters,
            "early_terminate": {"type": "hyperband", "min_iter": 2, "eta": 2},
        }

        project_name = f"{WANDB_PROJECT}-{EXP}-{task}-{USAR_BALANCE}"
        print(f"🔬 Iniciando W&B Sweep (bayes) | {n_trials} trials | proyecto: {project_name}")

        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, function=train_one_trial, count=n_trials)

    else:
        n_trials = N_TRIALS if N_TRIALS > 0 else (2 if TEST_MODE else 15)

        if TEST_MODE:
            print(f"⚠️  MODO PRUEBA: {n_trials} trials, 2 epochs, hiperparámetros fijos")

        print(f"🔬 Iniciando Optuna | {n_trials} trials | {len(train_groups)} grupos de entrenamiento...")

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
        )
        study.optimize(objective, n_trials=n_trials)

        print("-" * 80)
        print(f"🏆 MEJORES PARÁMETROS: {study.best_params}")
        print(f"📈 MEJOR SCORE GLOBAL: {study.best_value:.4f}")
import os
import json
import gc
import time

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
# CLASES
# ---------------------------------------------------------------------------

class EventTokenizer(nn.Module):
    def __init__(self, n_event_types, n_products, d_model,
                 use_balance=True,
                 event_emb_dim=64, prod_emb_dim=128, num_proj_dim=64):
        super().__init__()

        self.use_balance = use_balance

        # 1. Embeddings Categóricos
        self.event_emb = nn.Embedding(n_event_types + 1, event_emb_dim, padding_idx=0)
        self.prod_emb  = nn.Embedding(n_products + 1, prod_emb_dim, padding_idx=0)

        # 2. Proyección Numérica Dinámica
        input_num_dim = 3 if use_balance else 2
        self.num_projection = nn.Sequential(
            nn.Linear(input_num_dim, num_proj_dim),
            nn.SiLU(),
            nn.Linear(num_proj_dim, num_proj_dim)
        )

        # 3. Fusión Final
        total_input_dim = event_emb_dim + prod_emb_dim + num_proj_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )

    def forward(self, event_type, product_id, num_feats):
        if not self.use_balance:
            num_feats = num_feats[:, :, :2]

        e = self.event_emb(event_type)
        p = self.prod_emb(product_id)
        n = self.num_projection(num_feats)

        combined = torch.cat([e, p, n], dim=-1)
        return self.fusion(combined)


class MambaModel(nn.Module):
    def __init__(self, n_event_types, n_products, d_model=128, d_state=32, d_conv=4,
                 use_balance=True, event_emb_dim=64, prod_emb_dim=128, num_proj_dim=64):
        super().__init__()

        self.tokenizer = EventTokenizer(
            n_event_types=n_event_types,
            n_products=n_products,
            d_model=d_model,
            use_balance=use_balance,
            event_emb_dim=event_emb_dim,
            prod_emb_dim=prod_emb_dim,
            num_proj_dim=num_proj_dim
        )

        self.backbone = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_event_types + 1),
        )

        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, event_type, product_id, num_feats, return_embeddings=False):
        x = self.tokenizer(event_type, product_id, num_feats)
        h = self.backbone(x)
        logits = self.head(h)

        if return_embeddings:
            user_emb = self.contrastive_proj(h[:, -1, :])
            return logits, user_emb

        return logits

    def get_user_embedding(self, event_type, product_id, num_feats):
        self.eval()
        with torch.no_grad():
            x = self.tokenizer(event_type, product_id, num_feats)
            h = self.backbone(x)
            return h[:, -1, :]


class FastSequenceDataset(Dataset):
    def __init__(self, df, task="next_action"):
        self.task = task
        self.event_seqs  = df["INPUT_EVENT_SEQ"].to_list()
        self.prod_seqs   = df["INPUT_PRODUCT_SEQ"].to_list()
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


def get_dataloader_for_multiple_files(file_list, uids_to_keep, batch_size=512, task="both", num_workers=2):
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
        pin_memory=True, persistent_workers=False
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
# CONSTANTES DEL MODELO
# ---------------------------------------------------------------------------
N_PRODUCTS    = 440874
n_event_types = 21
d_model       = 128
d_state       = 32
d_conv        = 4

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
# MODELO PRINCIPAL
# ---------------------------------------------------------------------------
USAR_BALANCE = False

model = MambaModel(
    n_event_types=n_event_types,
    n_products=N_PRODUCTS,
    d_model=d_model,
    d_state=d_state,
    d_conv=d_conv,
    use_balance=USAR_BALANCE,
    event_emb_dim=32,
    prod_emb_dim=64,
    num_proj_dim=64
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
print(f"🚀 Modelo Mamba listo en {device}")

# ---------------------------------------------------------------------------
# SPLIT DE ARCHIVOS (GCS)
# ---------------------------------------------------------------------------
fs        = gcsfs.GCSFileSystem()
gcs_glob  = OUT_DIR.replace("gs://", "") + "/batch_*.parquet"
all_files = sorted(["gs://" + p for p in fs.glob(gcs_glob)])

if len(all_files) < 2:
    raise ValueError("Se necesitan al menos 2 archivos Parquet para tener train y val.")

print(f"Archivos encontrados: {len(all_files)}")

split_idx   = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files   = all_files[split_idx:]

# Agrupamos de 25 en 25 para no saturar RAM
train_groups = [train_files[i:i + 25] for i in range(0, len(train_files), 25)]
val_groups   = [val_files[i:i + 25]   for i in range(0, len(val_files),   25)]

print(f"🚀 MODO PRODUCCIÓN: Usando todos los archivos.")
print(f"📂 Archivos: {len(train_files)} Train | {len(val_files)} Val")


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


def upload_checkpoint_to_gcs(local_path: str, trial_number: int):
    """Sube un checkpoint local a GCS y elimina el archivo temporal."""
    gcs_path = GCS_CKPT_DIR.replace("gs://", "") + f"/best_model_trial_{trial_number}.ckpt"
    fs.put(local_path, gcs_path)
    os.remove(local_path)
    print(f"✅ Checkpoint subido a gs://{gcs_path}")


# ---------------------------------------------------------------------------
# FUNCIÓN OBJETIVO OPTUNA
# ---------------------------------------------------------------------------
def objective(trial):
    config = {
        "d_model":    trial.suggest_categorical("d_model", [128, 256, 512]),
        "d_state":    trial.suggest_int("d_state", 16, 64, step=16),
        "lambda_cl":  trial.suggest_float("lambda_cl", 0.1, 0.8),
        "lr":         trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "temp":       trial.suggest_float("temp", 0.05, 0.15),
        "batch_size": 128,
        "epochs":     20
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"trial-{trial.number}",
        config=config,
        reinit=True
    )

    model_t = MambaModel(
        n_products=N_PRODUCTS,
        n_event_types=21,
        d_model=config["d_model"],
        d_state=config["d_state"]
    ).to(device)

    optimizer_t    = torch.optim.AdamW(model_t.parameters(), lr=config["lr"])
    criterion_t    = nn.CrossEntropyLoss(ignore_index=0)
    early_stopper  = EarlyStopping(patience=3)
    best_trial_f1  = -1.0

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()

        # A. FASE ENTRENAMIENTO
        model_t.train()
        train_loss_accum = 0

        for group in train_groups:
            torch.cuda.empty_cache()

            loader = get_dataloader_for_multiple_files(
                group, train_uids, batch_size=config["batch_size"], task="both"
            )
            if loader is None:
                continue

            for batch in loader:
                optimizer_t.zero_grad()
                et     = batch["event_type"].to(device)
                pid    = batch["product_id"].to(device)
                nf     = batch["num_feats"].to(device)
                target = batch["target"].to(device)

                logits, emb1_raw = model_t(et, pid, nf, return_embeddings=True)
                et_v2            = apply_event_masking(et, mask_prob=0.15)
                _, emb2_raw      = model_t(et_v2, pid, nf, return_embeddings=True)

                emb1, emb2 = get_pooled_embedding(emb1_raw), get_pooled_embedding(emb2_raw)

                min_seq  = min(logits.shape[1], target.shape[1])
                loss_na  = criterion_t(
                    logits[:, :min_seq, :].reshape(-1, logits.size(-1)),
                    target[:, :min_seq].reshape(-1)
                )
                loss_cl    = compute_contrastive_loss(emb1, emb2, temp=config["temp"])
                loss_total = loss_na + (config["lambda_cl"] * loss_cl)
                loss_total.backward()
                optimizer_t.step()
                train_loss_accum += loss_total.item()

            del loader
            gc.collect()

        # B. FASE VALIDACIÓN
        model_t.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for v_group in val_groups:
                loader_val = get_dataloader_for_multiple_files(
                    v_group, test_uids, batch_size=config["batch_size"], task="next_action"
                )
                if loader_val is None:
                    continue
                for batch in loader_val:
                    et     = batch["event_type"].to(device)
                    pid    = batch["product_id"].to(device)
                    nf     = batch["num_feats"].to(device)
                    target = batch["target"].to(device)

                    logits = model_t(et, pid, nf)
                    preds  = torch.argmax(logits, dim=-1)

                    min_v = min(preds.shape[1], target.shape[1])
                    mask  = (target[:, :min_v] != 0)
                    all_targets.append(target[:, :min_v][mask].cpu())
                    all_preds.append(preds[:, :min_v][mask].cpu())
                del loader_val

        # C. MÉTRICAS Y RECURSOS
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()

        metrics_macro    = precision_recall_fscore_support(y_true, y_pred, average='macro',    zero_division=0)
        metrics_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        p_macro, r_macro, f1_m, _       = metrics_macro
        p_weighted, r_weighted, _, _     = metrics_weighted
        current_score                    = r_macro

        ram_usage      = psutil.Process().memory_info().rss / (1024 ** 3)
        gpu_usage      = torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        epoch_duration = time.time() - epoch_start_time

        print(
            f"Trial {trial.number} | Epoch {epoch + 1}/{config['epochs']} | "
            f"F1 Macro: {f1_m:.4f} | Recall Macro: {r_macro:.4f} | "
            f"RAM: {ram_usage:.2f} GB | GPU: {gpu_usage:.2f} GB | "
            f"Tiempo: {epoch_duration:.1f}s"
        )

        wandb.log({
            "epoch":                epoch + 1,
            "train_loss":           train_loss_accum / max(len(train_groups), 1),
            "val_recall_macro":     r_macro,
            "val_recall_weighted":  r_weighted,
            "val_precision_macro":  p_macro,
            "val_precision_weighted": p_weighted,
            "val_f1_macro":         f1_m,
            "ram_gb":               ram_usage,
            "gpu_gb":               gpu_usage,
            "epoch_duration_sec":   epoch_duration
        })

        if current_score > best_trial_f1:
            best_trial_f1 = current_score
            local_ckpt    = f"/tmp/best_model_trial_{trial.number}.ckpt"
            torch.save({
                "model_state_dict": model_t.state_dict(),
                "config":           config,
                "recall_macro":     r_macro
            }, local_ckpt)
            upload_checkpoint_to_gcs(local_ckpt, trial.number)

        trial.report(f1_m, epoch)
        if trial.should_prune():
            run.finish()
            raise optuna.exceptions.TrialPruned()

        early_stopper(f1_m)
        if early_stopper.early_stop:
            print(f"Early stopping en epoch {epoch + 1}")
            break

    run.finish()
    del model_t, optimizer_t
    torch.cuda.empty_cache()
    gc.collect()

    return best_trial_f1


# ---------------------------------------------------------------------------
# EJECUCIÓN DEL ESTUDIO OPTUNA
# ---------------------------------------------------------------------------
n_trials_to_run = 15

print(f"🔬 Iniciando estudio con {len(train_groups)} grupos de entrenamiento...")

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
)
study.optimize(objective, n_trials=n_trials_to_run)

print("-" * 80)
print(f"🏆 MEJORES PARÁMETROS: {study.best_params}")
print(f"📈 MEJOR F1 GLOBAL: {study.best_value:.4f}")
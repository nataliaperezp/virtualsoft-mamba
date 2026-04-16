"""
eval_contrastive_clusters.py
=============================
Evalúa la calidad de los embeddings contrastivos agrupando usuarios en clusters.

Uso (dentro del contenedor o con env vars seteadas):
  python3 eval_contrastive_clusters.py \\
      --checkpoint gs://ml-bucketvs/.../best_model_trial_2.ckpt \\
      --k 8 \\
      --out clusters.html

  python3 eval_contrastive_clusters.py \\
      --checkpoint gs://... \\
      --k-range 4,16 \\
      --max-files 10
"""

import argparse
import os
import tempfile

import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  plotly no instalado — se omitirá el HTML")

import gcsfs
import polars as pl

# Importamos solo lo necesario de Mamba.py
# (el guard __main__ evita que se ejecute el estudio Optuna al importar)
from Mamba import (
    MambaModel, FastSequenceDataset, collate_fn,
    OUT_DIR, UIDS_PATH,
    device, N_PRODUCTS,
    train_uids, test_uids,
    val_files as MAMBA_VAL_FILES,
)

# ---------------------------------------------------------------------------
# ARGUMENTOS
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  type=str, required=True,
                    help="Ruta GCS del checkpoint")
parser.add_argument("--k",           type=int, default=8)
parser.add_argument("--k-range",     type=str, default=None,
                    help="Buscar mejor k, ej: '4,16'")
parser.add_argument("--batch-size",  type=int, default=256)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--max-files",   type=int, default=None,
                    help="Limitar nº de parquets val (prueba rápida)")
parser.add_argument("--out",         type=str, default="clusters.html")
args = parser.parse_args()

fs = gcsfs.GCSFileSystem()

# ---------------------------------------------------------------------------
# CARGA DEL CHECKPOINT
# ---------------------------------------------------------------------------
print(f"\n📦 Cargando checkpoint: {args.checkpoint}")

with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
    tmp_path = f.name
fs.get(args.checkpoint.replace("gs://", ""), tmp_path)
ckpt = torch.load(tmp_path, map_location=device)
os.remove(tmp_path)

config       = ckpt["config"]
exp_info     = ckpt.get("experiment", {})
usar_balance = exp_info.get("usar_balance", False)

print(f"  d_model={config['d_model']} | d_state={config['d_state']} | "
      f"task={exp_info.get('task','?')} | trial={exp_info.get('trial','?')} | "
      f"epoch={exp_info.get('epoch','?')}")

model = MambaModel(
    n_products=N_PRODUCTS, n_event_types=21,
    d_model=config["d_model"], d_state=config["d_state"],
    use_balance=usar_balance
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("  ✅ Modelo cargado")

# ---------------------------------------------------------------------------
# ARCHIVOS VAL
# ---------------------------------------------------------------------------
val_files = MAMBA_VAL_FILES[:args.max_files] if args.max_files else MAMBA_VAL_FILES
print(f"\n📂 {len(val_files)} archivos val")

# ---------------------------------------------------------------------------
# GENERACIÓN DE EMBEDDINGS
# ---------------------------------------------------------------------------
print(f"\n🔄 Generando embeddings...")

uid_embeddings = defaultdict(list)
group_size     = 10
val_groups     = [val_files[i:i+group_size] for i in range(0, len(val_files), group_size)]

with torch.no_grad():
    for g_idx, group in enumerate(val_groups):
        df = pl.concat([pl.scan_parquet(f) for f in group]).filter(
            pl.col("USER_ID").is_in(list(test_uids))
        ).collect()

        if len(df) == 0:
            continue

        uids_grupo = df["USER_ID"].to_list()
        ds         = FastSequenceDataset(df, task="contrastive")
        loader     = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=args.num_workers,
                                pin_memory=args.num_workers > 0,
                                prefetch_factor=1 if args.num_workers > 0 else None)
        seq_idx = 0

        for batch in loader:
            et  = batch["event_type"].to(device)
            pid = batch["product_id"].to(device)
            nf  = batch["num_feats"].to(device)

            with torch.cuda.amp.autocast():
                emb = model.get_user_embedding(et, pid, nf)

            emb_np = emb.cpu().float().numpy()
            for i in range(emb_np.shape[0]):
                uid_embeddings[uids_grupo[seq_idx + i]].append(emb_np[i])
            seq_idx += emb_np.shape[0]

        print(f"  [{g_idx+1}/{len(val_groups)}] {len(uid_embeddings):,} usuarios")

# ---------------------------------------------------------------------------
# AGREGACIÓN POR USUARIO
# ---------------------------------------------------------------------------
uids       = list(uid_embeddings.keys())
emb_matrix = np.array([np.mean(uid_embeddings[u], axis=0) for u in uids])
emb_norm   = normalize(emb_matrix)
print(f"\n  {len(uids):,} usuarios × {emb_matrix.shape[1]} dims")

# ---------------------------------------------------------------------------
# BÚSQUEDA DEL MEJOR K
# ---------------------------------------------------------------------------
if args.k_range:
    k_min, k_max = map(int, args.k_range.split(","))
    print(f"\n🔍 Buscando mejor k en {list(range(k_min, k_max+1, 2))}...")
    scores = {}
    for k in range(k_min, k_max + 1, 2):
        labels_ = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3).fit_predict(emb_norm)
        score   = silhouette_score(emb_norm, labels_, sample_size=min(5000, len(uids)))
        scores[k] = score
        print(f"  k={k:2d} → silhouette={score:.4f}")
    args.k = max(scores, key=scores.get)
    print(f"  ✅ Mejor k={args.k}")

# ---------------------------------------------------------------------------
# CLUSTERING FINAL
# ---------------------------------------------------------------------------
print(f"\n🎯 K-Means con k={args.k}...")
labels    = KMeans(n_clusters=args.k, random_state=42, n_init=10).fit_predict(emb_norm)
sil_score = silhouette_score(emb_norm, labels, sample_size=min(10000, len(uids)))

print(f"  Silhouette: {sil_score:.4f}  (>0.5 excelente | 0.25-0.5 aceptable | <0.25 débil)")

unique, counts = np.unique(labels, return_counts=True)
print(f"\n  Distribución:")
for cl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
    bar = "█" * int(cnt / len(uids) * 50)
    print(f"    Cluster {cl:2d}: {cnt:6,} ({cnt/len(uids)*100:5.1f}%) {bar}")

# ---------------------------------------------------------------------------
# TOP EVENT IDs POR CLUSTER
# ---------------------------------------------------------------------------
print(f"\n🔬 Top 5 event IDs por cluster (muestra del primer grupo)...")
uid_to_cluster = dict(zip(uids, labels))
sample_df      = pl.concat([pl.scan_parquet(f) for f in val_files[:group_size]]).filter(
    pl.col("USER_ID").is_in(list(test_uids))
).collect()

cluster_event_counts = defaultdict(lambda: defaultdict(int))
for row in sample_df.iter_rows(named=True):
    uid = row["USER_ID"]
    if uid not in uid_to_cluster:
        continue
    for ev in row["INPUT_EVENT_SEQ"]:
        if ev > 0:
            cluster_event_counts[uid_to_cluster[uid]][ev] += 1

for cl in range(args.k):
    counts_cl = cluster_event_counts[cl]
    if not counts_cl:
        continue
    total   = sum(counts_cl.values())
    top5    = sorted(counts_cl.items(), key=lambda x: -x[1])[:5]
    top_str = " | ".join([f"ev_{ev} {cnt/total*100:.0f}%" for ev, cnt in top5])
    print(f"    Cluster {cl:2d}: {top_str}")

# ---------------------------------------------------------------------------
# VISUALIZACIÓN HTML
# ---------------------------------------------------------------------------
if HAS_PLOTLY:
    print(f"\n🎨 Generando visualización 2D...")
    pca   = PCA(n_components=2, random_state=42)
    emb2d = pca.fit_transform(emb_norm)
    var   = pca.explained_variance_ratio_.sum() * 100

    idx = np.random.choice(len(uids), min(5000, len(uids)), replace=False)
    fig = px.scatter(
        x=emb2d[idx, 0], y=emb2d[idx, 1],
        color=[str(labels[i]) for i in idx],
        hover_name=[str(uids[i]) for i in idx],
        title=f"Embeddings contrastivos | k={args.k} | silhouette={sil_score:.3f} | PCA var={var:.1f}%",
        labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
        width=900, height=650
    )
    fig.write_html(args.out)
    print(f"  ✅ Guardado en '{args.out}'")

print(f"""
{'='*50}
  Usuarios   : {len(uids):,}
  Clusters   : {args.k}
  Silhouette : {sil_score:.4f}
{'='*50}
""")

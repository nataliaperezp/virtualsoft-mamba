import os
import polars as pl
import gc
import time
import io  # <-- CRUCIAL: Necesario para el buffer de memoria
from google.cloud import storage
from dotenv import load_dotenv

# Cargar variables desde el archivo .env
load_dotenv()

# --- CONFIGURACIÓN GCS ---
# Usamos os.getenv para leer lo que configuraste en tu archivo .env
BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")
OUT_DIR_GCS = os.getenv("OUT_DIR")
CHUNK_DIR = os.getenv("CHUNK_DIR")

# Inicializar cliente de Storage
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)


def process_batch_all_history(events_lf, batch_ids, batch_idx, m_val, s_val, N_WINDOW, STRIDE, OUT_DIR_GCS):
    start_time = time.perf_counter()

    # 1. Ajuste de IDs y Filtrado
    lf = (
        events_lf
        .filter(pl.col("USER_ID").is_in(batch_ids))
        .select([
            pl.col("USER_ID").cast(pl.Int64),
            pl.col("TIMESTAMP_EVENT"),
            pl.col("VALUE").cast(pl.Float32),
            pl.col("EVENT_TYPE").cast(pl.Int8),
            pl.col("PRODUCT_ID").cast(pl.Int64).fill_null(0)
        ])
        .sort(["USER_ID", "TIMESTAMP_EVENT"])
        .with_columns([
            (pl.col("TIMESTAMP_EVENT").diff().over("USER_ID").dt.total_seconds().fill_null(0))
            .alias("TIME_DELTA")
        ])
    )

    # 2. Lógica de Balance y Normalización
    lf = lf.with_columns([
        pl.when(pl.col("EVENT_TYPE").is_in([2, 4])).then(-pl.col("VALUE"))
        .when(pl.col("EVENT_TYPE").is_in([3, 5, 19])).then(pl.col("VALUE"))
        .when(pl.col("EVENT_TYPE") == 21).then(-pl.col("VALUE"))
        .otherwise(0).cast(pl.Float32).alias("NET_CASHFLOW_T"),
    ]).with_columns([
        pl.col("NET_CASHFLOW_T").cum_sum().over("USER_ID").shift(1).fill_null(0).alias("BALANCE")
    ]).with_columns([
        ((pl.col("VALUE").fill_null(0) - m_val) / s_val).cast(pl.Float32).alias("VALUE_NORM"),
        (pl.col("BALANCE").abs().add(1).log()).cast(pl.Float32).alias("BALANCE_NORM"),
        (pl.col("TIME_DELTA").add(1).log()).cast(pl.Float32).alias("TIME_DELTA_NORM")
    ])

    # 3. Agregación
    df_agg = lf.group_by("USER_ID").agg([
        pl.col("EVENT_TYPE"),
        pl.col("PRODUCT_ID"),
        pl.struct([
            pl.col("VALUE_NORM"),
            pl.col("TIME_DELTA_NORM"),
            pl.col("BALANCE_NORM")
        ]).alias("NUM_FEATS")
    ]).collect()

    # 4. Sliding Windows
    final_rows = []
    for row in df_agg.iter_rows(named=True):
        u_id, e_type, p_id, n_feats = row["USER_ID"], row["EVENT_TYPE"], row["PRODUCT_ID"], row["NUM_FEATS"]
        n_events = len(e_type)
        n_windows = max(1, (n_events - N_WINDOW) // STRIDE + 1)

        for w in range(n_windows):
            start = w * STRIDE
            final_rows.append({
                "USER_ID": u_id,
                "INPUT_EVENT_SEQ": e_type[start: start + N_WINDOW],
                "INPUT_PRODUCT_SEQ": p_id[start: start + N_WINDOW],
                "NUM_FEATS_SEQ": n_feats[start: start + N_WINDOW],
                "TARGET_EVENT_SEQ": e_type[start + 1: start + N_WINDOW + 1]
            })

    # 5. Guardado DIRECTO AL BUCKET (Buffer en RAM)
    if final_rows:
        blob_name = f"{OUT_DIR_GCS}/batch_{batch_idx}_all_history.parquet"
        blob = bucket.blob(blob_name)

        buffer = io.BytesIO()
        pl.DataFrame(final_rows).write_parquet(buffer, compression="zstd")
        blob.upload_from_string(buffer.getvalue(), content_type="application/octet-stream")
        buffer.close()

    duration = time.perf_counter() - start_time
    del df_agg, final_rows
    gc.collect()
    return duration


# --- EJECUCIÓN ---
N_WINDOW, STRIDE = 1000, 500
GLOBAL_M_VAL, GLOBAL_S_VAL = 23.1213, 459.0598

archivos_chunks = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith('.parquet')])
print(f"📦 Total de chunks a procesar hacia GCP: {len(archivos_chunks)}")

for i, chunk_name in enumerate(archivos_chunks):
    chunk_done_blob = bucket.blob(f"{OUT_DIR_GCS}/DONE_{chunk_name}.txt")

    if chunk_done_blob.exists():
        print(f"⏭️ Saltando {chunk_name}, ya existe en GCP.")
        continue

    try:
        lf_chunk = pl.scan_parquet(os.path.join(CHUNK_DIR, chunk_name))
        ids = lf_chunk.select("USER_ID").unique().collect().get_column("USER_ID").to_list()

        STEP = 1000
        print(f"🔄 Procesando {chunk_name} ({len(ids)} usuarios)...")
        for j in range(0, len(ids), STEP):
            label = f"c{i}_b{j}"
            process_batch_all_history(lf_chunk, ids[j:j + STEP], label,
                                      GLOBAL_M_VAL, GLOBAL_S_VAL, N_WINDOW, STRIDE, OUT_DIR_GCS)

        # Marcar como finalizado en el bucket
        chunk_done_blob.upload_from_string("finalizado con éxito")
        gc.collect()

    except Exception as e:
        print(f"❌ Error crítico en chunk {chunk_name}: {e}")

print(f"\n✨ ¡PROCESAMIENTO TERMINADO! Revisa gs://{BUCKET_NAME}/{OUT_DIR_GCS}/")
"""
launch_experiments.py — Lanzador de experimentos para Vertex AI
================================================================
Uso:
  python3 launch_experiments.py                  # muestra el grid y pregunta cuáles lanzar
  python3 launch_experiments.py --list           # solo lista los experimentos
  python3 launch_experiments.py --run 1,3,5      # lanza los experimentos 1, 3 y 5
  python3 launch_experiments.py --run all        # lanza todos
  python3 launch_experiments.py --dry-run --run 2  # imprime el comando sin ejecutar
  python3 launch_experiments.py --test --run 1   # prueba rápida (~1000 usuarios, 2 trials)
  python3 launch_experiments.py --trials 5 --run all  # 5 trials de Optuna para todos

  # Modo estático (sin Optuna, hiperparámetros fijos):
  python3 launch_experiments.py --static --run 5 \
      --d-model 256 --d-state 64 --lr 5.23e-4 --lambda-cl 0.5 --temp 0.106 --epochs 20
"""

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime

# ---------------------------------------------------------------------------
# GRID DE EXPERIMENTOS
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {"id": 1, "task": "next_action",  "usar_balance": False, "desc": "Next-action sin balance"},
    {"id": 2, "task": "next_action",  "usar_balance": True,  "desc": "Next-action con balance"},
    {"id": 3, "task": "contrastive",  "usar_balance": False, "desc": "Contrastivo sin balance"},
    {"id": 4, "task": "contrastive",  "usar_balance": True,  "desc": "Contrastivo con balance"},
    {"id": 5, "task": "both",         "usar_balance": False, "desc": "Next-action + Contrastivo sin balance"},
    {"id": 6, "task": "both",         "usar_balance": True,  "desc": "Next-action + Contrastivo con balance"},
]

# ---------------------------------------------------------------------------
# CONFIGURACIÓN GCP
# ---------------------------------------------------------------------------
WORKERS = 10
BATCH_GROUP_SIZE = 25
BATCH_SIZE = 512
EPOCHS = 30
ESTADO= "Estático"
PROJECT_ID    = "composed-arch-276322"
REGION        = "us-east1"
IMAGE         = f"gcr.io/{PROJECT_ID}/mamba-trainer-contrastive:v3"
MACHINE_TYPE  = "g2-standard-16"
ACCELERATOR   = "NVIDIA_L4"
OUT_DIR       = "gs://ml-bucketvs/mamba-exp/data/sequences_output_v3"
CKPT_DIR      = f"gs://ml-bucketvs/mamba-exp/checkpoints/{ACCELERATOR}_gs{BATCH_GROUP_SIZE}_w{WORKERS}_bs{BATCH_SIZE}_region_{REGION}_{EPOCHS}_{ESTADO}"
WANDB_PROJECT = f"mamba-train-{BATCH_GROUP_SIZE}-w{WORKERS }_bs{BATCH_SIZE}_region_{REGION}_{EPOCHS}_{ESTADO}"
UIDS_PATH     = "/app/split_estratificado_uids.json"
WANDB_API_KEY = os.environ.get(
    "WANDB_API_KEY",
    "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"
)
#{'d_model': 512, 'd_state': 16, 'lambda_cl': 0.733838076306787, 'lr': 0.00011019269288754933, 'temp': 0.12576519146387083}
def print_grid():
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│               GRID DE EXPERIMENTOS MAMBA                   │")
    print("├────┬─────────────────┬───────────────┬─────────────────────┤")
    print("│ ID │ task            │ usar_balance  │ descripción         │")
    print("├────┼─────────────────┼───────────────┼─────────────────────┤")
    for exp in EXPERIMENTS:
        balance_str = "Sí" if exp["usar_balance"] else "No"
        print(f"│ {exp['id']:2} │ {exp['task']:<15} │ {balance_str:<13} │ {exp['desc']:<19} │")
    print("└────┴─────────────────┴───────────────┴─────────────────────┘\n")


def build_job_yaml(exp: dict, test_mode: bool = False, test_files: int = 10,
                   group_size: int = 0, n_epochs: int = 0,
                   batch_size: int = 0, num_workers: int = 0,
                   n_trials: int = 0,
                   static_mode: bool = False, d_model: int = 256, d_state: int = 64,
                   lr: float = 5e-4, lambda_cl: float = 0.5, temp: float = 0.1) -> str:
    """Genera el contenido YAML del job para --config."""
    test_suffix = "-test" if test_mode else ""
    static_suffix = "-static" if static_mode else ""
    balance_val = "True" if exp["usar_balance"] else "False"
    ckpt_dir    = f"{CKPT_DIR}/exp{exp['id']}{test_suffix}{static_suffix}"

    # Resolver defaults aquí para que el YAML sea siempre explícito y auditable
    resolved_epochs      = n_epochs     if n_epochs     > 0 else (2   if test_mode else 10)
    resolved_batch_size  = batch_size   if batch_size   > 0 else (64  if test_mode else 512)
    resolved_group_size  = group_size   if group_size   > 0 else (2   if test_mode else 100)
    resolved_num_workers = num_workers  if num_workers  > 0 else 6
    resolved_trials      = n_trials     if n_trials     > 0 else (2   if test_mode else 15)

    env_vars = [
        {"name": "OUT_DIR",                    "value": OUT_DIR},
        {"name": "GCS_CKPT_DIR",               "value": ckpt_dir},
        {"name": "WANDB_PROJECT",              "value": WANDB_PROJECT},
        {"name": "UIDS_PATH",                  "value": UIDS_PATH},
        {"name": "WANDB_API_KEY",              "value": WANDB_API_KEY},
        {"name": "EXP",                        "value": str(exp["id"])},
        {"name": "TASK",                       "value": exp["task"]},
        {"name": "USAR_BALANCE",               "value": balance_val},
        {"name": "PYTORCH_CUDA_ALLOC_CONF",    "value": "expandable_segments:True"},
        {"name": "N_EPOCHS",                   "value": str(resolved_epochs)},
        {"name": "BATCH_SIZE",                 "value": str(resolved_batch_size)},
        {"name": "GROUP_SIZE",                 "value": str(resolved_group_size)},
        {"name": "NUM_WORKERS",                "value": str(resolved_num_workers)},
    ]

    if static_mode:
        env_vars += [
            {"name": "STATIC_MODE",       "value": "True"},
            {"name": "STATIC_D_MODEL",    "value": str(d_model)},
            {"name": "STATIC_D_STATE",    "value": str(d_state)},
            {"name": "STATIC_LR",         "value": str(lr)},
            {"name": "STATIC_LAMBDA_CL",  "value": str(lambda_cl)},
            {"name": "STATIC_TEMP",       "value": str(temp)},
        ]
    else:
        env_vars.append({"name": "N_TRIALS", "value": str(resolved_trials)})

    if test_mode:
        env_vars.append({"name": "TEST_MODE",  "value": "True"})
        env_vars.append({"name": "TEST_FILES", "value": str(test_files)})

    env_lines = "\n".join(
        f"      - name: {e['name']}\n        value: \"{e['value']}\""
        for e in env_vars
    )

    return (
        "workerPoolSpecs:\n"
        "- machineSpec:\n"
        f"    machineType: {MACHINE_TYPE}\n"
        f"    acceleratorType: {ACCELERATOR}\n"
        "    acceleratorCount: 1\n"
        "  replicaCount: 1\n"
        "  containerSpec:\n"
        f"    imageUri: {IMAGE}\n"
        "    env:\n"
        f"{env_lines}\n"
    )

def launch_experiment(exp: dict, dry_run: bool = False, test_mode: bool = False,
                      test_files: int = 10, group_size: int = 0, n_epochs: int = 0,
                      batch_size: int = 0, num_workers: int = 0, n_trials: int = 0,
                      static_mode: bool = False, d_model: int = 256, d_state: int = 64,
                      lr: float = 5e-4, lambda_cl: float = 0.5, temp: float = 0.1):
    timestamp    = datetime.now().strftime("%Y%m%d-%H%M")
    test_suffix  = "-test" if test_mode else ""
    static_suffix = "-static" if static_mode else ""
    bs_tag       = f"bs{batch_size}" if batch_size > 0 else "bs512"
    machine_tag  = MACHINE_TYPE.replace("-", "").replace("standard", "s")  # g2s16
    job_name     = f"mamba-exp{exp['id']}-{exp['task']}{test_suffix}{static_suffix}-{bs_tag}-{machine_tag}-{timestamp}"
    balance_str = "con balance" if exp["usar_balance"] else "sin balance"

    yaml_content = build_job_yaml(exp, test_mode=test_mode, test_files=test_files,
                                  group_size=group_size, n_epochs=n_epochs,
                                  batch_size=batch_size, num_workers=num_workers,
                                  n_trials=n_trials, static_mode=static_mode,
                                  d_model=d_model, d_state=d_state,
                                  lr=lr, lambda_cl=lambda_cl, temp=temp)

    print(f"\n🚀 Lanzando Experimento {exp['id']}: {exp['task']} {balance_str}")
    print(f"   Descripción: {exp['desc']}")

    if dry_run:
        print(f"   [DRY-RUN] YAML que se usaría (--display-name={job_name}):")
        print(textwrap.indent(yaml_content, "   "))
        return

    # Escribimos el YAML en un archivo temporal y lanzamos con --config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        cmd = [
            "gcloud", "ai", "custom-jobs", "create",
            f"--region={REGION}",
            f"--display-name={job_name}",
            f"--project={PROJECT_ID}",
            f"--config={tmp_path}",
            "--labels=model=mamba-training",
        ]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"   ✅ Job lanzado correctamente")
        else:
            print(f"   ❌ Error al lanzar el job (código {result.returncode})")
            sys.exit(1)
    finally:
        os.remove(tmp_path)


def parse_run_ids(run_arg: str) -> list:
    if run_arg.strip().lower() == "all":
        return [exp["id"] for exp in EXPERIMENTS]
    ids = []
    for part in run_arg.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
        else:
            print(f"❌ ID inválido: '{part}'. Usa números separados por comas o 'all'.")
            sys.exit(1)
    return ids


def interactive_selection() -> list:
    print_grid()
    print("¿Cuáles experimentos quieres lanzar?")
    print("  Escribe los IDs separados por comas (ej: 1,3,5) o 'all' para todos:")
    raw = input("  > ").strip()
    if not raw:
        print("No se seleccionó ningún experimento. Saliendo.")
        sys.exit(0)
    return parse_run_ids(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Lanzador de experimentos Mamba en Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Ejemplos:
              python3 launch_experiments.py                         # modo interactivo
              python3 launch_experiments.py --list                  # lista experimentos
              python3 launch_experiments.py --run 1,3               # lanza exp 1 y 3
              python3 launch_experiments.py --run all               # lanza todos
              python3 launch_experiments.py --dry-run --run 2       # sin ejecutar (muestra el YAML)
              python3 launch_experiments.py --test --run 1          # prueba rápida (~1000 usuarios, 2 trials)
              python3 launch_experiments.py --test --test-files 5 --run 1
        """)
    )
    parser.add_argument("--list",       action="store_true", help="Lista el grid de experimentos y sale")
    parser.add_argument("--run",        type=str, default=None, help="IDs a lanzar (ej: 1,3,5 o 'all')")
    parser.add_argument("--dry-run",    action="store_true", help="Muestra el YAML sin ejecutar")
    parser.add_argument("--test",        action="store_true", help="Modo prueba: pocos datos, 2 trials, 2 epochs")
    parser.add_argument("--test-files",  type=int, default=10,
                        help="Nro. de archivos parquet en modo prueba (default: 10 ≈ 1000 usuarios)")
    parser.add_argument("--group-size",   type=int, default=0,
                        help="Archivos por grupo de carga (default: 2 prueba / 100 prod)")
    parser.add_argument("--epochs",       type=int, default=0,
                        help="Épocas de entrenamiento (default: 2 prueba / 10 prod)")
    parser.add_argument("--batch-size",   type=int, default=0,
                        help="Batch size (default: 64 prueba / 512 prod)")
    parser.add_argument("--num-workers",  type=int, default=0,
                        help="Workers del DataLoader (default: 6)")
    parser.add_argument("--trials",       type=int, default=0,
                        help="Trials de Optuna (default: 2 prueba / 15 prod)")
    # Modo estático
    parser.add_argument("--static",       action="store_true",
                        help="Modo estático: entrena con hiperparámetros fijos, sin Optuna")
    parser.add_argument("--d-model",      type=int,   default=256,   help="d_model (modo estático, default: 256)")
    parser.add_argument("--d-state",      type=int,   default=64,    help="d_state (modo estático, default: 64)")
    parser.add_argument("--lr",           type=float, default=5e-4,  help="learning rate (modo estático, default: 5e-4)")
    parser.add_argument("--lambda-cl",    type=float, default=0.5,   help="lambda contrastive (modo estático, default: 0.5)")
    parser.add_argument("--temp",         type=float, default=0.1,   help="temperatura contrastive (modo estático, default: 0.1)")
    args = parser.parse_args()

    if args.list:
        print_grid()
        return

    exp_map = {exp["id"]: exp for exp in EXPERIMENTS}

    if args.run:
        selected_ids = parse_run_ids(args.run)
    else:
        selected_ids = interactive_selection()

    invalid = [i for i in selected_ids if i not in exp_map]
    if invalid:
        print(f"❌ IDs no existentes: {invalid}. Experimentos válidos: {list(exp_map.keys())}")
        sys.exit(1)

    selected_exps = [exp_map[i] for i in selected_ids]
    dry_label  = " [DRY-RUN]"   if args.dry_run else ""
    test_label = " [TEST MODE]" if args.test    else ""

    print(f"\n📋 Se lanzarán {len(selected_exps)} experimento(s){dry_label}{test_label}:")
    for exp in selected_exps:
        balance_str = "con balance" if exp["usar_balance"] else "sin balance"
        print(f"   • Exp {exp['id']}: {exp['task']} {balance_str} — {exp['desc']}")

    test_mode = args.test
    eff_epochs      = args.epochs     if args.epochs     > 0 else (2   if test_mode else 10)
    eff_batch_size  = args.batch_size if args.batch_size > 0 else (64  if test_mode else 512)
    eff_group_size  = args.group_size if args.group_size > 0 else (2   if test_mode else 100)
    eff_trials      = args.trials     if args.trials     > 0 else (2   if test_mode else 15)

    if args.static:
        print(f"\n   Config estática: d_model={args.d_model} | d_state={args.d_state} | lr={args.lr:.2e} | lambda_cl={args.lambda_cl:.2f} | temp={args.temp:.3f}", end="")
    else:
        print(f"\n   Config efectiva: epochs={eff_epochs} | batch_size={eff_batch_size} | group_size={eff_group_size} | trials={eff_trials}", end="")
    if test_mode:
        print(f" | test_files={args.test_files} [PRUEBA]", end="")
    print()

    if not args.dry_run and not args.run:
        confirm = input("\n¿Confirmar lanzamiento? [s/N]: ").strip().lower()
        if confirm not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado.")
            return

    for exp in selected_exps:
        launch_experiment(exp, dry_run=args.dry_run, test_mode=args.test,
                          test_files=args.test_files, group_size=args.group_size,
                          n_epochs=args.epochs, batch_size=args.batch_size,
                          num_workers=args.num_workers, n_trials=args.trials,
                          static_mode=args.static, d_model=args.d_model,
                          d_state=args.d_state, lr=args.lr,
                          lambda_cl=args.lambda_cl, temp=args.temp)

    if not args.dry_run:
        print(f"\n✅ {len(selected_exps)} job(s) enviados a Vertex AI.")
        print(f"   Monitorea en: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


if __name__ == "__main__":
    main()

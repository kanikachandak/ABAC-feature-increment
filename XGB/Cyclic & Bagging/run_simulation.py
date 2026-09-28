"""
run_simulation.py  –  Master experiment runner (merged-CSV version).
=====================================================================
Iterates over dataset × strategy × prep_type × train_method × seed and calls:
    python simulate_one.py --dataset company --dataset-csv company.csv ...

All results from every run are appended to ONE consolidated CSV:
    results/consolidated_results.csv

Usage
─────
  # Full run (3 datasets × 3 strategies × 4 prep-types × 2 methods × 1 seed)
  python run_simulation.py

  # Quick subset
  python run_simulation.py --datasets company --strategies sbp --prep-types 2 --seeds 0

  # Dry run
  python run_simulation.py --dry-run

  # Skip completed runs
  python run_simulation.py --skip-existing
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--datasets",      default="company,university1,university2")
parser.add_argument("--strategies",    default="sbp,pbp,rnp")
parser.add_argument("--prep-types",    default="2,3,4,5",
                    help="1=Naive 2=ARFE 3=AVC 4=ARFE+AVC 5=NaiveNA")
parser.add_argument("--train-methods", default="bagging,cyclic",
                    help="bagging and/or cyclic")
parser.add_argument("--seeds",         default="0")
parser.add_argument("--num-clients",   type=int, default=4)
parser.add_argument("--num-rounds",    type=int, default=50)
parser.add_argument("--local-epochs",  type=int, default=3)
parser.add_argument("--output-dir",    default="results")
parser.add_argument("--log-file",      default="experiment_run.log")
parser.add_argument("--dry-run",       action="store_true")
parser.add_argument("--skip-existing", action="store_true",
                    help="Skip if this run's rows already exist in consolidated CSV")
parser.add_argument("--fail-fast",     action="store_true")
parser.add_argument("--verbose",       action="store_true")
parser.add_argument("--shared-secret", default="",
                    help="AES-256-GCM shared secret; same on server and all clients")
args = parser.parse_args()

DATASETS  = [d.strip() for d in args.datasets.split(",")      if d.strip()]
STRATEGIES= [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
PREP_TYPES= [int(p) for p in args.prep_types.split(",")        if p.strip()]
METHODS   = [m.strip().lower() for m in args.train_methods.split(",") if m.strip()]
SEEDS     = [int(s) for s in args.seeds.split(",")             if s.strip()]

PREP_NAMES= {1:"Naive", 2:"ARFE", 3:"AVC", 4:"ARFE_AVC", 5:"NaiveNA"}
HERE      = Path(__file__).parent.resolve()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(args.log_file, mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _csv_path(dataset: str) -> Path | None:
    """Find merged CSV: look in project root then Dataset/ subfolder."""
    for base in [HERE, HERE / "Dataset"]:
        p = base / f"{dataset}.csv"
        if p.exists():
            return p
    return None


def _already_done(dataset, method, prep_type, partition, seed) -> bool:
    """Check consolidated CSV for this exact run's final round row."""
    csv_path = HERE / args.output_dir / "consolidated_results.csv"
    if not csv_path.exists():
        return False
    import csv
    prep = PREP_NAMES.get(prep_type, str(prep_type))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("dataset")            == dataset and
                row.get("train_method")        == method  and
                row.get("prep_type")           == prep    and
                row.get("partition_strategy")  == partition and
                str(row.get("seed",""))        == str(seed)):
                return True
    return False


def _simulate_script() -> Path:
    s = HERE / "simulate_one.py"
    if not s.exists():
        log.error("simulate_one.py not found in %s", HERE)
        sys.exit(1)
    return s


simulate_script = _simulate_script()

# ─── Validate CSVs ───────────────────────────────────────────────────────────
if not args.dry_run:
    for ds in DATASETS:
        p = _csv_path(ds)
        if p is None:
            log.error("'%s.csv' not found in %s (or %s/Dataset/)", ds, HERE, HERE)
            sys.exit(1)
        log.info("Dataset '%s' → %s", ds, p.name)


# ─── Build experiment list ────────────────────────────────────────────────────
experiments = [
    {"dataset": ds, "strategy": st, "prep_type": pt, "method": me, "seed": sd}
    for ds in DATASETS
    for st in STRATEGIES
    for pt in PREP_TYPES
    for me in METHODS
    for sd in SEEDS
]
total = len(experiments)

log.info("\n%s", "=" * 60)
log.info("  EXPERIMENT PLAN")
log.info("  Datasets   : %s", DATASETS)
log.info("  Strategies : %s", [s.upper() for s in STRATEGIES])
log.info("  Prep types : %s", [PREP_NAMES.get(p,p) for p in PREP_TYPES])
log.info("  Methods    : %s", METHODS)
log.info("  Seeds      : %s", SEEDS)
log.info("  Clients    : %d  |  Rounds: %d", args.num_clients, args.num_rounds)
log.info("  Total runs : %d", total)
log.info("  Results    : %s/consolidated_results.csv", args.output_dir)
log.info("%s\n", "=" * 60)


# ─── Run one experiment ───────────────────────────────────────────────────────

def run_one(exp: dict, run_num: int) -> bool:
    dataset   = exp["dataset"]
    strategy  = exp["strategy"]
    prep_type = exp["prep_type"]
    method    = exp["method"]
    seed      = exp["seed"]
    prep_name = PREP_NAMES.get(prep_type, str(prep_type))
    label     = (f"[{run_num}/{total}] {dataset} / {strategy.upper()} / "
                 f"{prep_name} / {method} / seed={seed}")

    if args.skip_existing and _already_done(dataset, method, prep_type, strategy, seed):
        log.info("%s  →  SKIPPED", label)
        return True

    csv_p   = _csv_path(dataset)
    out_dir = str(HERE / args.output_dir)

    # ── Per-experiment hyperparameter tuning ─────────────────────────────────
    # AVC (prep_type=3) and ARFE_AVC (prep_type=4) collapse fine-grained
    # attribute values into coarse groups, reducing the effective feature space
    # dramatically.  XGBoost needs more capacity and more rounds to find
    # decision boundaries in this coarser space.
    #
    # University2 (35:1 imbalance) also needs careful tuning regardless of encoding.
    #
    # ┌─────────────────────────┬──────┬────────┬──────┬────────────┬──────────┐
    # │ Case                    │rounds│max_depth│ eta │min_child_w │ spw_power│
    # ├─────────────────────────┼──────┼────────┼──────┼────────────┼──────────┤
    # │ AVC/ARFE_AVC + uni2     │ 200  │   10   │ 0.01 │     2      │  0.33    │
    # │ AVC/ARFE_AVC + others   │ 150  │    8   │ 0.02 │     3      │  0.33    │
    # │ ARFE/NaiveNA + uni2     │ 100  │    6   │ 0.03 │     5      │  0.50    │
    # │ ARFE/NaiveNA + others   │  50  │    6   │ 0.05 │     5      │  0.50    │
    # └─────────────────────────┴──────┴────────┴──────┴────────────┴──────────┘
    #
    # spw_power controls scale_pos_weight = ratio^spw_power:
    #   0.50 → sqrt(ratio): balanced for ARFE/NaiveNA
    #   0.33 → cbrt(ratio): reduces false-positive rate for coarse AVC features,
    #           improving precision at the cost of slight recall reduction

    is_avc  = prep_type in [3, 4]
    is_uni2 = dataset == "university2"

    if is_avc and is_uni2:                       # hardest case
        num_rounds, max_depth = 200, 10
        eta, min_cw           = 0.01, 2
        local_epochs, spw_pwr = 5,    0.33
    elif is_avc:                                  # AVC, non-uni2
        num_rounds, max_depth = 150, 8
        eta, min_cw           = 0.02, 3
        local_epochs, spw_pwr = 4,    0.33
    elif is_uni2:                                 # ARFE/NaiveNA, uni2
        num_rounds, max_depth = 100, 6
        eta, min_cw           = 0.03, 5
        local_epochs, spw_pwr = 3,    0.50
    else:                                         # ARFE/NaiveNA, others
        num_rounds, max_depth = args.num_rounds, 6
        eta, min_cw           = 0.05, 5
        local_epochs, spw_pwr = args.local_epochs, 0.50

    cmd = [
        sys.executable, str(simulate_script),
        "--dataset",           dataset,
        "--strategy",          strategy,
        "--prep-type",         str(prep_type),
        "--seed",              str(seed),
        "--num-clients",       str(args.num_clients),
        "--num-rounds",        str(num_rounds),
        "--local-epochs",      str(local_epochs),
        "--train-method",      method,
        "--dataset-csv",       str(csv_p),
        "--output-dir",        out_dir,
        "--max-depth",         str(max_depth),
        "--eta",               str(eta),
        "--min-child-weight",  str(min_cw),
        "--subsample",         "0.8",
        "--colsample-bytree",  "0.8",
        "--spw-power",         str(spw_pwr),
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.shared_secret:
        cmd += ["--shared-secret", args.shared_secret]

    if args.dry_run:
        log.info("DRY-RUN  %s\n  CMD: %s", label, " ".join(cmd))
        return True

    log.info("\n%s\n%s\n%s", "─" * 60, label, "─" * 60)
    t0 = time.perf_counter()
    try:
        result  = subprocess.run(cmd, cwd=str(HERE))
        elapsed = time.perf_counter() - t0
        if result.returncode == 0:
            log.info("%s  →  SUCCESS (%.1fs)", label, elapsed)
            return True
        else:
            log.error("%s  →  FAILED (exit=%d, %.1fs)",
                      label, result.returncode, elapsed)
            return False
    except Exception as exc:
        log.error("%s  →  EXCEPTION: %s", label, exc)
        return False


# ─── Main loop ────────────────────────────────────────────────────────────────

start_all = time.perf_counter()
results   = []

for i, exp in enumerate(experiments, start=1):
    ok = run_one(exp, i)
    results.append((exp, ok))
    if not ok and args.fail_fast:
        log.error("--fail-fast: stopping.")
        break

elapsed_total = time.perf_counter() - start_all
passed  = sum(1 for _, ok in results if ok)
failed  = len(results) - passed

log.info("\n%s", "=" * 60)
log.info("  SUMMARY  Total=%d  Passed=%d  Failed=%d  Time=%s",
         total, passed, failed, str(timedelta(seconds=int(elapsed_total))))
log.info("  Results → %s/consolidated_results.csv", args.output_dir)
log.info("%s", "=" * 60)

if failed:
    log.info("\n  FAILED:")
    for exp, ok in results:
        if not ok:
            log.info("    %s / %s / %s / %s / seed=%s",
                     exp["dataset"], exp["strategy"],
                     PREP_NAMES.get(exp["prep_type"], exp["prep_type"]),
                     exp["method"], exp["seed"])
    sys.exit(1)

log.info("All experiments completed successfully.")

"""
simulate_one.py  –  Run ONE federated experiment via Flower's Python API.
=========================================================================
Uses a single merged CSV per dataset (company.csv / university1.csv /
university2.csv). Splits it 80/20 into train/test internally.

Called by run_simulation.py:
    python simulate_one.py --dataset company --strategy sbp --prep-type 2 ...

Avoids `flwr run` / `python -m flwr` which both fail with:
    "No module named flwr.__main__"
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# ─── CLI ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--dataset",       required=True, choices=["company","university1","university2"])
p.add_argument("--strategy",      required=True, choices=["sbp","pbp","rnp"])
p.add_argument("--prep-type",     required=True, type=int, choices=[1,2,3,4,5])
p.add_argument("--seed",          required=True, type=int)
p.add_argument("--num-clients",   default=4,     type=int)
p.add_argument("--num-rounds",    default=50,    type=int)
p.add_argument("--local-epochs",  default=3,     type=int)
p.add_argument("--train-method",  default="bagging", choices=["bagging","cyclic"])
p.add_argument("--dataset-csv",   required=True, help="Path to merged CSV")
p.add_argument("--output-dir",    default="results")
p.add_argument("--verbose",       action="store_true")
# ── Per-experiment XGBoost hyperparameters ───────────────────────────────────
p.add_argument("--max-depth",         type=int,   default=6,    help="XGBoost max_depth")
p.add_argument("--eta",               type=float, default=0.05, help="XGBoost learning rate")
p.add_argument("--min-child-weight",  type=int,   default=5,    help="XGBoost min_child_weight")
p.add_argument("--subsample",         type=float, default=0.8,  help="XGBoost subsample")
p.add_argument("--colsample-bytree",  type=float, default=0.8,  help="XGBoost colsample_bytree")
p.add_argument("--spw-power",         type=float, default=0.5,
               help="Exponent for scale_pos_weight: 0.5=sqrt (default), 0.33=cbrt (AVC)")
p.add_argument("--shared-secret",     default="",
               help="AES-256-GCM shared secret (FL_SHARED_SECRET); same on server+clients")
args = p.parse_args()

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("simulate_one")

PREP_NAMES = {1:"Naive", 2:"ARFE", 3:"AVC", 4:"ARFE_AVC", 5:"NaiveNA"}

# ─── Set experiment parameters as environment variables ──────────────────────
os.environ["DATASET_NAME"]       = args.dataset
os.environ["PARTITION_STRATEGY"] = args.strategy
os.environ["PREP_TYPE"]          = str(args.prep_type)
os.environ["TRAIN_METHOD"]       = args.train_method
os.environ["FLWR_SEED"]          = str(args.seed)
os.environ["PYTHONHASHSEED"]     = str(args.seed)
os.environ["DATASET_CSV"]        = os.path.abspath(args.dataset_csv)
os.environ["OUTPUT_DIR"]         = os.path.abspath(args.output_dir)
os.environ["CENTRALISED_EVAL"]   = "true"
os.environ["NUM_ROUNDS"]         = str(args.num_rounds)
os.environ["LOCAL_EPOCHS"]       = str(args.local_epochs)
os.environ["MAX_DEPTH"]          = str(args.max_depth)
os.environ["ETA"]                = str(args.eta)
os.environ["MIN_CHILD_WEIGHT"]   = str(args.min_child_weight)
os.environ["SUBSAMPLE"]          = str(args.subsample)
os.environ["COLSAMPLE_BYTREE"]   = str(args.colsample_bytree)
os.environ["SPW_POWER"]          = str(args.spw_power)
if args.shared_secret:
    os.environ["FL_SHARED_SECRET"] = args.shared_secret

os.makedirs(args.output_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)

log.info("=" * 60)
log.info("Dataset=%-12s  Strategy=%-4s  PrepType=%d(%s)  Seed=%d",
         args.dataset, args.strategy.upper(),
         args.prep_type, PREP_NAMES[args.prep_type], args.seed)
log.info("Clients=%d  Rounds=%d  LocalEpochs=%d  Method=%s",
         args.num_clients, args.num_rounds, args.local_epochs, args.train_method)
log.info("CSV=%s", args.dataset_csv)
log.info("=" * 60)

# ─── Import apps AFTER env vars are set ──────────────────────────────────────
from server import app as server_app  # noqa: E402
from client import app as client_app  # noqa: E402
from flwr.simulation import run_simulation  # noqa: E402

log.info("Starting simulation  num_supernodes=%d ...", args.num_clients)

run_simulation(
    server_app     = server_app,
    client_app     = client_app,
    num_supernodes = args.num_clients,
    backend_name   = "ray",
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    verbose_logging= args.verbose,
)

log.info("Done.  Results → %s/consolidated_results.csv", args.output_dir)

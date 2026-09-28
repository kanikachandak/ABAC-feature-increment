import argparse
import importlib
import inspect
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Experiment grid configuration
# ---------------------------------------------------------------------------

DATASET_DISPLAY_NAMES = {
    "company": "Company",
    "university1": "University1",
    "university2": "University2",
}

# Folder name -> partition code expected by your paper/metrics naming
PARTITION_CODES = {
    "SBC": "SBP",    # Scoring-Based Partition
    "PBP": "PBP",  # Perception-Based Partition
    "RNP": "RNP",  # Random Partition
}

# prep_type -> encoding name
ENCODING_NAMES = {
    2: "ARFE",
    3: "AVC",
    4: "ARFE_AVC",
    5: "Naive_NACol",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_csv_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for x in _parse_csv_list(raw):
        out.append(int(x))
    return out


def _parse_seeds(raw: str) -> list[int]:
    """Parse seeds like '0', '0,1,2', '0-9'."""
    raw = (raw or "").strip()
    if not raw:
        return [0]
    if "-" in raw and "," not in raw:
        a, b = raw.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if b < a:
            a, b = b, a
        return list(range(a, b + 1))
    return _parse_int_list(raw)


def _ensure_empty_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


@contextmanager
def _temp_sys_path(path: Path):
    """Temporarily add a directory to sys.path at highest priority."""
    path_str = str(path.resolve())
    old_sys_path = list(sys.path)
    try:
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        yield
    finally:
        sys.path[:] = old_sys_path


def _purge_modules(module_names: list[str]):
    for name in module_names:
        if name in sys.modules:
            del sys.modules[name]
    importlib.invalidate_caches()


def _copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _patch_job_configs(job_dir: Path, data_dir: Path, test_processed_path: Path):
    """Patch NVFlare job configs to point at this case's prepared data and test set."""
    app_dir = job_dir / "app"
    if not app_dir.exists():
        raise FileNotFoundError(f"Job folder missing app/: {job_dir}")

    # Patch every server config we find (config/, config1/, etc.)
    for server_cfg in app_dir.glob("config*/config_fed_server*.json"):
        cfg = _read_json(server_cfg)
        comps = cfg.get("components", [])
        for c in comps:
            args = c.get("args", {})
            # Server evaluator (any component that has a test_data_path arg)
            if "test_data_path" in args:
                args["test_data_path"] = str(test_processed_path.resolve())
        _write_json(server_cfg, cfg)

    # Patch every client config we find
    for client_cfg in app_dir.glob("config*/config_fed_client*.json"):
        cfg = _read_json(client_cfg)
        comps = cfg.get("components", [])
        for c in comps:
            args = c.get("args", {})
            # CSVDataLoader (any component that has a folder arg)
            if "folder" in args:
                args["folder"] = str(data_dir.resolve())
        _write_json(client_cfg, cfg)


def _get_simulator_runner():
    """Import NVFlare only when needed, so --prepare-only can run without it."""
    try:
        from nvflare import SimulatorRunner  # type: ignore
        return SimulatorRunner
    except Exception:
        return None


def _call_prepare_data_split(case_dir: Path, prep_type: int, seed: int, n_clients: int, valid_rows: int):
    """Import the case-local prepare_data.py and run prepare_data_split."""
    with _temp_sys_path(case_dir):
        # IMPORTANT: each case has a different data_preprocessor.py.
        # We must purge module caches or Python will re-use previous imports.
        _purge_modules(["data_preprocessor", "prepare_data"])

        import prepare_data  # noqa: F401
        func = getattr(prepare_data, "prepare_data_split", None)
        if func is None:
            raise AttributeError(f"prepare_data_split not found in {case_dir}/prepare_data.py")

        sig = inspect.signature(func)
        kwargs: dict = {}

        # Positional 1st param is prep_type in all your variants
        args = [prep_type]

        if "seed" in sig.parameters:
            kwargs["seed"] = seed
        if "n_clients" in sig.parameters:
            kwargs["n_clients"] = n_clients
        if "require_valid_placeholder" in sig.parameters:
            kwargs["require_valid_placeholder"] = True
        if "valid_rows" in sig.parameters:
            kwargs["valid_rows"] = valid_rows

        func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Case staging
# ---------------------------------------------------------------------------

def _resolve_train_test(project_root: Path, dataset: str, allow_root_fallback: bool) -> tuple[Path, Path]:
    """Find train.csv/test.csv for a dataset.

    Preferred layout (recommended):
      <project_root>/<dataset>/train.csv
      <project_root>/<dataset>/test.csv

    Optional legacy fallback (only if allow_root_fallback=True):
      <project_root>/train.csv
      <project_root>/test.csv
    """
    dataset_dir = project_root / dataset

    cand_train = dataset_dir / "train.csv"
    cand_test = dataset_dir / "test.csv"
    if cand_train.exists() and cand_test.exists():
        return cand_train, cand_test

    if allow_root_fallback:
        root_train = project_root / "train.csv"
        root_test = project_root / "test.csv"
        if root_train.exists() and root_test.exists():
            print(
                f"WARNING: Using fallback train/test from project root for dataset '{dataset}'.\n"
                f"  train: {root_train}\n  test:  {root_test}\n"
                "  (Recommended: place train.csv/test.csv inside the dataset folder.)"
            )
            return root_train, root_test

    raise FileNotFoundError(
        f"Could not find train.csv/test.csv for dataset '{dataset}'.\n"
        f"Expected: {cand_train} and {cand_test}"
    )


def _stage_case_files(
    project_root: Path,
    dataset: str,
    strategy_folder: str,
    case_dir: Path,
    allow_root_fallback: bool,
) -> dict:
    """Create a self-contained case directory with the right scripts + data."""
    dataset_dir = project_root / dataset
    scripts_dir = dataset_dir / strategy_folder
    if not scripts_dir.exists():
        raise FileNotFoundError(f"Missing scripts dir: {scripts_dir}")

    train_src, test_src = _resolve_train_test(project_root, dataset, allow_root_fallback)

    _ensure_empty_dir(case_dir)

    # Copy the strategy-specific code
    _copy_file(scripts_dir / "data_preprocessor.py", case_dir / "data_preprocessor.py")
    _copy_file(scripts_dir / "prepare_data.py", case_dir / "prepare_data.py")

    # Copy train/test into the case folder (so prepare_data.py finds them next to __file__)
    _copy_file(train_src, case_dir / "train.csv")
    _copy_file(test_src, case_dir / "test.csv")

    # Pre-create output dirs
    (case_dir / "results").mkdir(parents=True, exist_ok=True)
    (case_dir / "models").mkdir(parents=True, exist_ok=True)
    (case_dir / "workspace").mkdir(parents=True, exist_ok=True)

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "scripts_dir": str(scripts_dir.resolve()),
        "train_src": str(train_src.resolve()),
        "test_src": str(test_src.resolve()),
    }


def _stage_job(project_root: Path, case_dir: Path) -> Path:
    """Copy the NVFlare job template into the case dir."""
    template_job_dir = project_root / "jobs" / "secure_project"
    if not template_job_dir.exists():
        raise FileNotFoundError(f"Missing job template: {template_job_dir}")

    dest_job_dir = case_dir / "job" / "secure_project"
    if dest_job_dir.exists():
        shutil.rmtree(dest_job_dir)
    dest_job_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template_job_dir, dest_job_dir)

    return dest_job_dir


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiments(
    project_root: Path,
    datasets: list[str],
    strategies: list[str],
    prep_types: list[int],
    seeds: list[int],
    n_clients: int,
    valid_rows: int,
    results_root: Path,
    prepare_only: bool,
    allow_root_fallback: bool,
):
    SimulatorRunner = _get_simulator_runner()
    if not prepare_only and SimulatorRunner is None:
        raise ImportError(
            "NVFlare is not installed in this Python environment.\n"
            "Install nvflare (and deps) or run with --prepare-only."
        )

    results_root.mkdir(parents=True, exist_ok=True)
    all_case_frames: list[pd.DataFrame] = []

    for dataset in datasets:
        if dataset not in DATASET_DISPLAY_NAMES:
            raise ValueError(f"Unknown dataset key: {dataset}. Expected one of {list(DATASET_DISPLAY_NAMES)}")

        dataset_display = DATASET_DISPLAY_NAMES[dataset]

        for strategy in strategies:
            if strategy not in PARTITION_CODES:
                raise ValueError(f"Unknown strategy folder: {strategy}. Expected one of {list(PARTITION_CODES)}")

            partition_code = PARTITION_CODES[strategy]

            for prep_type in prep_types:
                if prep_type not in ENCODING_NAMES:
                    raise ValueError(f"Unsupported prep_type: {prep_type}. Expected one of {list(ENCODING_NAMES)}")

                encoding_name = ENCODING_NAMES[prep_type]
                case_name = f"{dataset_display}_{partition_code}_{encoding_name}"

                case_dir = results_root / dataset / partition_code / encoding_name
                print("\n" + "=" * 88)
                print(f"CASE: {case_name}")
                print(f"Results dir: {case_dir}")
                print("=" * 88)

                # 1) Stage scripts + data into a self-contained case folder
                staging_info = _stage_case_files(
                    project_root=project_root,
                    dataset=dataset,
                    strategy_folder=strategy,
                    case_dir=case_dir,
                    allow_root_fallback=allow_root_fallback,
                )

                # 2) Stage job template and patch it for this case
                job_dir = _stage_job(project_root, case_dir)

                # Paths that will be created by prepare_data_split
                data_dir = case_dir / "data"
                test_processed_path = case_dir / "test_processed.csv"

                # Patch job configs so server/client read from this case directory
                _patch_job_configs(job_dir, data_dir=data_dir, test_processed_path=test_processed_path)

                # 3) Run across seeds
                wall_times: dict[int, float] = {}

                # metrics file written by server_eval.py
                metrics_csv_path = case_dir / "results" / "server_metrics.csv"

                for seed in seeds:
                    print("\n" + "-" * 70)
                    print(f"Seed {seed} | prep_type={prep_type} ({encoding_name}) | partition={partition_code}")
                    print("-" * 70)

                    # Export env vars consumed by server_eval.py (and optionally trainers)
                    os.environ["EXPERIMENT_SEED"] = str(seed)
                    os.environ["XGB_SEED"] = str(seed)
                    os.environ["PREP_TYPE"] = str(prep_type)
                    os.environ["PARTITION"] = partition_code
                    os.environ["DATASET"] = dataset_display
                    os.environ["ENCODING"] = encoding_name
                    os.environ["CASE_NAME"] = case_name
                    os.environ["METRICS_GROUP"] = f"prep{prep_type}"
                    os.environ["METRICS_CSV_PATH"] = str(metrics_csv_path.resolve())
                    os.environ["MODEL_SAVE_DIR"] = str((case_dir / "models").resolve())

                    # Data prep (creates data/site-*/train.csv + test_processed.csv)
                    _call_prepare_data_split(case_dir, prep_type=prep_type, seed=seed, n_clients=n_clients, valid_rows=valid_rows)

                    if prepare_only:
                        print("[prepare-only] Skipping NVFlare simulation run.")
                        wall_times[seed] = 0.0
                        continue

                    # Use a seed-scoped workspace to avoid cross-seed collisions
                    seed_workspace = case_dir / "workspace" / f"seed_{seed}"
                    seed_workspace.mkdir(parents=True, exist_ok=True)

                    simulator = SimulatorRunner(
                        job_folder=str(job_dir),
                        workspace=str(seed_workspace),
                        n_clients=n_clients,
                        threads=n_clients,
                    )

                    t0 = time.perf_counter()
                    simulator.run()
                    t1 = time.perf_counter()

                    wall = t1 - t0
                    wall_times[seed] = wall
                    print(f"[Seed {seed}] Simulation wall time: {wall:.3f}s")

                # 4) Persist run metadata
                meta = {
                    "case_name": case_name,
                    "dataset": dataset,
                    "dataset_display": dataset_display,
                    "strategy_folder": strategy,
                    "partition": partition_code,
                    "prep_type": prep_type,
                    "encoding": encoding_name,
                    "n_clients": n_clients,
                    "seeds": seeds,
                    "staging": staging_info,
                    "job_dir": str(job_dir.resolve()),
                }
                (case_dir / "results" / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                (case_dir / "results" / "wall_times.json").write_text(json.dumps(wall_times, indent=2), encoding="utf-8")

                # 5) Create a convenient per-case final_metrics.csv (join metrics + wall time)
                if metrics_csv_path.exists():
                    df = pd.read_csv(metrics_csv_path)
                    if "seed" in df.columns:
                        df["_seed_int"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
                        df["wall_time_s"] = df["_seed_int"].map(lambda s: wall_times.get(int(s), None) if pd.notna(s) else None)
                        df = df.drop(columns=["_seed_int"])
                    out_path = case_dir / "results" / "final_metrics.csv"
                    df.to_csv(out_path, index=False)
                    print(f"Saved per-case metrics to: {out_path}")
                    all_case_frames.append(df)
                else:
                    print(f"WARNING: metrics CSV not found (simulation may have failed or was skipped): {metrics_csv_path}")

    # 6) Aggregate across all cases
    if all_case_frames:
        df_all = pd.concat(all_case_frames, ignore_index=True)
        out_all = results_root / "all_final_metrics.csv"
        df_all.to_csv(out_all, index=False)
        print("\n" + "=" * 88)
        print(f"Saved aggregated metrics CSV: {out_all}")
        print("=" * 88)


def main():
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Batch runner for NVFlare XGBoost experiments.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="company,university1,university2",
        help="Comma-separated datasets to run. Options: company,university1,university2",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="SBC,PBP,RNP",
        help="Comma-separated strategy folders to run. Options: SBC,PBP,RNP",
    )
    parser.add_argument(
        "--prep-types",
        type=str,
        default="2,3,4,5",
        help="Comma-separated prep types (encoding modes). Options: 2,3,4,5",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Seed spec: '0', '0,1,2', or '0-9'",
    )
    parser.add_argument("--n-clients", type=int, default=4)
    parser.add_argument("--valid-rows", type=int, default=50, help="Rows to write into valid.csv placeholder per client.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(project_root / "results"),
        help="Where to create case folders + store outputs.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only stage files + run prepare_data_split (no NVFlare training). Useful to validate the grid.",
    )
    parser.add_argument(
        "--allow-root-data-fallback",
        action="store_true",
        help="If a dataset folder is missing train.csv/test.csv, fall back to project-root train/test (legacy layout).",
    )

    args = parser.parse_args()

    datasets = _parse_csv_list(args.datasets)
    strategies = _parse_csv_list(args.strategies)
    prep_types = _parse_int_list(args.prep_types)
    seeds = _parse_seeds(args.seeds)

    run_experiments(
        project_root=project_root,
        datasets=datasets,
        strategies=strategies,
        prep_types=prep_types,
        seeds=seeds,
        n_clients=args.n_clients,
        valid_rows=args.valid_rows,
        results_root=Path(args.results_root),
        prepare_only=args.prepare_only,
        allow_root_fallback=args.allow_root_data_fallback,
    )


if __name__ == "__main__":
    main()

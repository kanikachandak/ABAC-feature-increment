# utils.py  –  Unified data utilities for all datasets.
# Accepts a SINGLE merged CSV per dataset (company.csv / university1.csv /
# university2.csv) and splits it 80/20 into train/test internally.
#
# Dataset behaviour is driven by the DATASET_NAME environment variable.
# Partition strategy is driven by the PARTITION_STRATEGY environment variable.

import os
import pandas as pd
import xgboost as xgb
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from dataset_registry import get_dataset_config


# ─────────────────────────── helpers ────────────────────────────────────────

def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().contiguous().numpy()
    return np.asarray(arr)


def _is_string_col(series: pd.Series) -> bool:
    """Works with both pandas 2.x (object) and 3.x (StringDtype)."""
    return not pd.api.types.is_numeric_dtype(series.dtype)


def same_conditions(col1, col2):
    if (col1 == -1) or (col2 == -1):
        return 2
    return 1 if col1 == col2 else 0


def chk_nota(col):
    return 1 if col == -1 else 0


def replace_keys(input_dict, match="-", target="_"):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        new_dict[new_key] = (replace_keys(value, match, target)
                             if isinstance(value, dict) else value)
    return new_dict


def compute_scale_pos_weight(y) -> float:
    """Return neg/pos ratio for XGBoost scale_pos_weight (handles class imbalance)."""
    labels = _to_numpy(y).ravel()
    n_neg  = float((labels == 0).sum())
    n_pos  = float((labels == 1).sum())
    return round(n_neg / n_pos, 4) if n_pos > 0 else 1.0


# ─────────────────────────── attribute mapping ──────────────────────────────

def GetAttributeMapping(data, cfg, grp_gap=20, map_type=1):
    mapping = {}
    mapping[cfg["nota_label"]] = -1
    mapping[0]                  =  0
    mapping[cfg["yes_label"]]   =  1
    mapping[cfg["no_label"]]    =  0

    t1s, t1e = cfg["mapping_t1_start"], cfg["mapping_t1_end"]
    t2s, t2e = cfg["mapping_t2_start"], cfg["mapping_t2_end"]
    attr_grp  = cfg["attr_grp"]

    if map_type == 1:
        for col in data.columns[t1s:t1e]:
            cnt = 1
            for val in data[col].unique():
                if val != cfg["nota_label"]:
                    mapping[val] = cnt
                    cnt += 1
        return mapping

    elif map_type == 2:
        for col in data.columns[t2s:t2e]:
            cnt = 1
            for val in data[col].unique():
                if val != cfg["nota_label"]:
                    mapping[val] = cnt
                    cnt += 1
        for grp_num, (g, members) in enumerate(attr_grp.items(), start=1):
            for mem_num, member in enumerate(members, start=1):
                for mem_val_num, val in enumerate(member, start=1):
                    mapping[val] = grp_num * grp_gap + mem_val_num
        return mapping

    raise ValueError(f"Unknown map_type: {map_type}")


# ─────────────────────────── data preparation ───────────────────────────────

def GetPreparedData(dataset_csv, prep_type=2, seed=0, k_best=1, cfg=None):
    """Load a single merged CSV, encode it, split 80/20, return train/test.

    Parameters
    ----------
    dataset_csv : str   path to the merged CSV (e.g. company.csv)
    prep_type   : int   1=Naive  2=ARFE  3=AVC  4=ARFE+AVC  5=Naive+NA
    seed        : int   random seed for reproducible train/test split
    k_best      : int   how many top features to rank via SelectKBest
    cfg         : dict  dataset config from dataset_registry (auto-loaded if None)

    Returns
    -------
    X_train, X_test, y_train, y_test,
    best_feature_index, best_feature_ordering, mapping
    """
    if cfg is None:
        cfg = get_dataset_config(os.environ.get("DATASET_NAME", "company"))

    data = pd.read_csv(dataset_csv)

    # ── feature ordering via SelectKBest on label-encoded copy ───────────────
    data_enc_tmp = data.copy()
    for col in data_enc_tmp.columns:
        if _is_string_col(data_enc_tmp[col]):
            le = LabelEncoder()
            data_enc_tmp[col] = le.fit_transform(data_enc_tmp[col].astype(str))

    X_tmp = data_enc_tmp.drop(columns=["Access"])
    y_tmp = data_enc_tmp["Access"]
    selector = SelectKBest(score_func=f_classif,
                           k=min(k_best, X_tmp.shape[1]))
    selector.fit(StandardScaler().fit_transform(X_tmp), y_tmp)
    feat_order = [f for f, _ in sorted(
        zip(X_tmp.columns, selector.scores_),
        key=lambda t: t[1], reverse=True
    )]
    print("Feature ordering (best→worst):", feat_order)

    # ── encoding by prep_type ────────────────────────────────────────────────
    if prep_type == 1:
        mapping      = GetAttributeMapping(data, cfg, map_type=1)
        data_encoded = data.replace(mapping).infer_objects()

    elif prep_type == 2:
        mapping      = GetAttributeMapping(data, cfg, map_type=1)
        data_encoded = data.replace(mapping).infer_objects()
        for (c1, c2, new_col) in cfg["arfe_pairs"]:
            if c1 in data_encoded.columns and c2 in data_encoded.columns:
                data_encoded[new_col] = data_encoded.apply(
                    lambda r, a=c1, b=c2: same_conditions(r[a], r[b]), axis=1)
        data_encoded = data_encoded.drop(
            columns=[c for c in cfg["arfe_drop"] if c in data_encoded.columns])

    elif prep_type == 3:
        mapping      = GetAttributeMapping(data, cfg, map_type=2)
        data_encoded = data.replace(mapping).infer_objects()

    elif prep_type == 4:
        mapping      = GetAttributeMapping(data, cfg, map_type=2)
        data_encoded = data.replace(mapping).infer_objects()
        for (c1, c2, new_col) in cfg["arfe_pairs"]:
            if c1 in data_encoded.columns and c2 in data_encoded.columns:
                data_encoded[new_col] = data_encoded.apply(
                    lambda r, a=c1, b=c2: same_conditions(r[a], r[b]), axis=1)
        data_encoded = data_encoded.drop(
            columns=[c for c in cfg["arfe_drop"] if c in data_encoded.columns])

    elif prep_type == 5:
        mapping      = GetAttributeMapping(data, cfg, map_type=1)
        data_encoded = data.replace(mapping).infer_objects()
        for (src_col, na_col) in cfg["na_cols"]:
            if src_col in data_encoded.columns:
                data_encoded[na_col] = data_encoded[src_col].apply(chk_nota)

    else:
        raise ValueError(f"Unknown prep_type: {prep_type}")

    # ── split X / y, find best feature index inside X ────────────────────────
    X = data_encoded.drop(columns=["Access"])
    y = data_encoded[["Access"]]

    best_feature_index = -1
    for feat in feat_order:
        if feat in X.columns:
            best_feature_index = list(X.columns).index(feat)
            break

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test, best_feature_index, feat_order, mapping


# ─────────────────────────── partitioning ───────────────────────────────────

def partition_dataset(X_train, y_train, best_feature, best_feature_name,
                      n_clients, partition_strategy="sbp", seed=42):
    """Partition training data among n_clients using RNP / PBP / SBP."""
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(_to_numpy(X_train), dtype=torch.float32)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(_to_numpy(y_train), dtype=torch.float32).view(-1, 1)

    rng = np.random.default_rng(int(seed))
    n   = X_train.shape[0]

    # ── RNP: round-robin random ──────────────────────────────────────────────
    if partition_strategy.lower() == "rnp":
        perm    = rng.permutation(n)
        X_parts = [[] for _ in range(n_clients)]
        y_parts = [[] for _ in range(n_clients)]
        for i, idx in enumerate(perm):
            cid = i % n_clients
            X_parts[cid].append(X_train[idx])
            y_parts[cid].append(y_train[idx])

        X_out, y_out, counts = [], [], []
        for cid in range(n_clients):
            Xc = torch.stack(X_parts[cid]) if X_parts[cid] else X_train[:0]
            yc = torch.stack(y_parts[cid]) if y_parts[cid] else y_train[:0]
            X_out.append(Xc); y_out.append(yc)
            counts.append((int((yc == 1).sum()), int((yc == 0).sum())))
        return X_out, y_out, counts

    # ── PBP / SBP: cluster by feature, round-robin within cluster ───────────
    feat_col    = X_train[:, best_feature]
    unique_vals = sorted(torch.unique(feat_col).tolist(), key=str)

    X_parts = [[] for _ in range(n_clients)]
    y_parts = [[] for _ in range(n_clients)]

    for val in unique_vals:
        idx = (feat_col == val).nonzero(as_tuple=True)[0].cpu().numpy()
        rng.shuffle(idx)
        for i in range(n_clients):
            cli_idx = idx[i::n_clients]
            if len(cli_idx) == 0:
                continue
            ti = torch.tensor(cli_idx, dtype=torch.long)
            X_parts[i].append(X_train[ti])
            y_parts[i].append(y_train[ti])

    X_out, y_out, counts = [], [], []
    for i in range(n_clients):
        Xc = torch.cat(X_parts[i], dim=0) if X_parts[i] else X_train[:0]
        yc = torch.cat(y_parts[i], dim=0) if y_parts[i] else y_train[:0]
        X_out.append(Xc); y_out.append(yc)
        counts.append((int((yc == 1).sum()), int((yc == 0).sum())))

    return X_out, y_out, counts


# ─────────────────────────── load_data (called by client) ───────────────────

def load_data(partition_id, num_partitions, seed=42,
              dataset_csv="dataset.csv",
              Partition_Type=2, target_column="Access"):
    """Main entry-point for the Flower client.

    Reads DATASET_NAME and PARTITION_STRATEGY from the environment.

    Returns
    -------
    train_dmatrix, valid_dmatrix, num_train, num_val, scale_pos_weight
    """
    dataset_name       = os.environ.get("DATASET_NAME", "company")
    partition_strategy = os.environ.get("PARTITION_STRATEGY", "sbp").lower()
    seed               = int(seed)
    cfg                = get_dataset_config(dataset_name)

    X_tr_df, X_te_df, y_tr_df, y_te_df, best_feat_idx, feat_order, _ = \
        GetPreparedData(dataset_csv, Partition_Type, seed, k_best=1, cfg=cfg)

    cols = list(X_tr_df.columns)

    # ── choose partitioning feature ──────────────────────────────────────────
    if partition_strategy == "rnp":
        best_feat_idx  = 0
        best_feat_name = "RNP"

    elif partition_strategy == "pbp":
        pbp_feat = os.environ.get("PBP_FEATURE", cfg["pbp_feature"])
        if pbp_feat in cols:
            best_feat_idx  = cols.index(pbp_feat)
            best_feat_name = pbp_feat
        else:
            best_feat_idx  = cfg["pbp_fallback_index"]
            best_feat_name = cols[best_feat_idx] if cols else "PBP_FALLBACK"

    else:  # sbp
        if best_feat_idx == -1:
            best_feat_idx  = 0
            best_feat_name = "No best feature"
        else:
            best_feat_name = cols[best_feat_idx]

    spw = compute_scale_pos_weight(y_tr_df)
    print(f"[load_data] dataset={dataset_name} strategy={partition_strategy} "
          f"prep={Partition_Type} partition={partition_id}/{num_partitions} "
          f"best_feat='{best_feat_name}'(idx={best_feat_idx}) spw={spw}")

    X_train = torch.tensor(X_tr_df.values, dtype=torch.float32)
    X_test  = torch.tensor(X_te_df.values, dtype=torch.float32)
    y_train = torch.tensor(y_tr_df.values, dtype=torch.float32).view(-1, 1)
    y_test  = torch.tensor(y_te_df.values, dtype=torch.float32).view(-1, 1)

    X_clients, y_clients, _ = partition_dataset(
        X_train, y_train,
        best_feat_idx, best_feat_name,
        num_partitions,
        partition_strategy=partition_strategy,
        seed=seed,
    )

    Xp = X_clients[partition_id]
    yp = y_clients[partition_id]

    train_dmatrix = xgb.DMatrix(_to_numpy(Xp),     label=_to_numpy(yp).ravel())
    valid_dmatrix = xgb.DMatrix(_to_numpy(X_test), label=_to_numpy(y_test).ravel())

    return train_dmatrix, valid_dmatrix, int(Xp.shape[0]), int(X_test.shape[0]), spw

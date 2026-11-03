"""
PBP_Company.py  –  Partition By Best-feature clustering federated learning
==========================================================================
Runs all 4 prep_types (ARFE=2, AVC=3, ARFE+AVC=4, NaiveNA=5).

Policy-enforcement artefacts (Scaler, Mapping, raw TestData CSVs)
have been REMOVED.  Only the trained model weights are persisted.

╔══════════════════════════════════════════════════════════╗
║  MODES                                                   ║
╠══════════════════════════════════════════════════════════╣
║  multithread  (default)                                  ║
║    Simulates 4 client threads + server on ONE machine.   ║
║    Uses torch.multiprocessing.spawn; world_size = 4.     ║
║    Run:  python PBP_Company.py                           ║
║                                                          ║
║  prep                                                    ║
║    Data-preparation step for multimachine mode.          ║
║    Run once on the machine that has the dataset.         ║
║    Saves encoded tensors (.pt) to Tensors/ folder.       ║
║    Run:  python PBP_Company.py --mode prep               ║
║          python PBP_Company.py --mode prep --prep_type 2 ║
║                                                          ║
║  multimachine                                            ║
║    Real distributed: one process per physical VM.        ║
║    world_size = 5  (rank 0 = global server, 1-4 = FL     ║
║    clients). Communicates over TCP via Gloo backend.     ║
║                                                          ║
║    Prerequisites: run 'prep' mode first so that encoded  ║
║    tensor files exist on shared / mounted storage.       ║
║                                                          ║
║    On server VM (rank 0):                                ║
║      python PBP_Company.py --mode multimachine \\        ║
║             --rank 0 --master_addr <SERVER_IP> \\        ║
║             --master_port 29502 --prep_type 2            ║
║    On each client VM (ranks 1-4):                        ║
║      python PBP_Company.py --mode multimachine \\        ║
║             --rank <R> --master_addr <SERVER_IP> \\      ║
║             --master_port 29502 --prep_type 2            ║
╚══════════════════════════════════════════════════════════╝

Artefacts saved
───────────────
  Model/PBP/<prep_name>/ArtificialNeuralNetwork_final.pth
  Results/Company/PBP/<prep_name>/Seed_0/results.csv
  consolidated_results.csv   ← 12-case summary across all algos/prep_types

Changes from baseline
─────────────────────
 - Added old/current feature-set data loading, preprocessing, and PBP partitioning.
 - Added separate old/current ANN models and parameter initialization for the new feature.
 - Added per-sample weights with balancing, normalization, and model regularization.
 - Added alternating optimization of old/current models and sample weights with FedAvg.
"""

import argparse
import csv
import os
import time
from datetime import timedelta
import math

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ALGO          = "PBP"
MODEL_ROOT    = os.path.join("Model", "feat-incr")
N_CLIENTS     = 4
WORLD_SIZE_MM = 5      # 1 server (rank 0) + 4 clients (ranks 1-4)
MT_PORT       = 7777   # localhost port for multithread TCPStore

# consistency lambdas
LAMBDA_U = 1e-4
LAMBDA_U_NORM = 1e-2
LAMBDA_BAL = 1e-4
LAMBDA_CON = 1e-4
LAMBDA_W = 1e-4

# ─── Perception-Based Partition (PBP) feature ────────────────────────────────
# PBP is *perception-based*: the partition feature is chosen by domain
# perception, NOT by an automatic SelectKBest score (that was the SBC/scoring
# variant).  We resolve it by NAME so it stays valid across all prep_types even
# when some columns are dropped.  Switch to "Department" if your perception
# feature is the subject department instead of designation.
BEST_FEATURE_NAME = "DESIGNATION"   # Subject Designation (reference perception)
NEW_FEATURE_NAMES = {"WORK_LOCATION"}


def resolve_best_feature(columns):
    """Return (index, name) of the perception feature within the encoded X."""
    cols = list(columns)
    if BEST_FEATURE_NAME in cols:
        return cols.index(BEST_FEATURE_NAME), BEST_FEATURE_NAME
    # Fallback: first column (keeps the run alive if the name was dropped)
    return 0, cols[0]


def resolve_feature_indices(old_columns, current_columns):
    """Map old attributes to their positions in the current feature matrix."""
    old_keys = _column_keys(old_columns)
    current_keys = _column_keys(current_columns)
    current_positions = {key: index for index, key in enumerate(current_keys)}
    missing = [key for key in old_keys if key not in current_positions]
    if missing:
        raise ValueError(f"Current dataset is missing old attributes: {missing}")
    old_indices = [current_positions[key] for key in old_keys]
    old_key_set = set(old_keys)
    new_indices = [index for index, key in enumerate(current_keys)
                   if key not in old_key_set]
    if not new_indices:
        raise ValueError("Current dataset does not contain a new feature")
    return old_indices, new_indices


def resolve_checkpoint_feature_indices(current_columns, old_columns,
                                       old_input_dim,
                                       new_feature_names=NEW_FEATURE_NAMES):
    """Locate inserted feature columns and validate the baseline width.

    Baseline checkpoints in this repository predate schema metadata, so the
    processed old schema is checked against the expanded schema.  This both
    identifies the explicitly named new attribute and prevents a silent
    permutation of old columns.
    """
    current_keys = _column_keys(current_columns)
    old_keys = _column_keys(old_columns)
    new_indices = [index for index, (name, _) in enumerate(current_keys)
                   if name in new_feature_names]
    old_indices = [index for index in range(len(current_keys))
                   if index not in set(new_indices)]
    if not new_indices:
        raise ValueError(
            f"No new feature found. Expected one of {sorted(new_feature_names)}"
        )
    if len(old_indices) != old_input_dim or len(old_keys) != old_input_dim:
        raise ValueError(
            f"Old schema/checkpoint width mismatch: old schema={len(old_keys)}, "
            f"expanded old columns={len(old_indices)}, checkpoint={old_input_dim}."
        )
    expanded_old_keys = [current_keys[index] for index in old_indices]
    if expanded_old_keys != old_keys:
        raise ValueError(
            "Expanded dataset old-column order does not match the baseline "
            f"schema. Baseline={old_keys}; expanded={expanded_old_keys}."
        )
    return old_indices, new_indices


def load_baseline_state(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    required = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    missing = required.difference(checkpoint.keys())
    if missing:
        raise ValueError(f"Baseline checkpoint is missing parameters: {sorted(missing)}")
    state = {key: checkpoint[key].detach().cpu() for key in required}
    if state["fc1.weight"].ndim != 2 or state["fc1.bias"].shape != (state["fc1.weight"].shape[0],):
        raise ValueError("Baseline checkpoint has an invalid fc1 architecture")
    if state["fc2.weight"].shape != (1, state["fc1.weight"].shape[0]):
        raise ValueError("Baseline checkpoint has an incompatible fc2 architecture")
    if state["fc2.bias"].shape != (1,):
        raise ValueError("Baseline checkpoint has an invalid fc2 bias")
    return state


# ─── Model definitions ────────────────────────────────────────────────────────

class ArtificialNeuralNetwork(nn.Module):
    """Expanded ANN with only the new-feature first-layer connections trainable.

    ``fc1`` and ``fc2`` contain the frozen baseline model.  The trainable
    parameter is an explicit delta for the columns identified by
    ``new_feature_indices``.  This avoids trying to freeze individual slices of
    a single PyTorch Parameter.
    """

    def __init__(self, baseline_state, input_dim, new_feature_indices,
                 learning_rate=0.001, max_iter=200):
        super().__init__()
        old_weight = baseline_state["fc1.weight"]
        hidden_size, old_input_dim = old_weight.shape
        if len(new_feature_indices) + old_input_dim != input_dim:
            raise ValueError(
                f"Checkpoint input width is {old_input_dim}, expanded data "
                f"width is {input_dim}, and {len(new_feature_indices)} new "
                "feature columns were identified."
            )

        self.input_dim = input_dim
        self.hidden_layer_size = hidden_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.criterion = nn.BCELoss(reduction="none")
        self.new_feature_indices = torch.tensor(
            list(new_feature_indices), dtype=torch.long
        )

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        expanded_weight = torch.zeros(
            hidden_size, input_dim, dtype=old_weight.dtype
        )
        old_indices = [i for i in range(input_dim)
                       if i not in set(new_feature_indices)]
        expanded_weight[:, old_indices] = old_weight
        self.fc1.weight.data.copy_(expanded_weight)
        self.fc1.bias.data.copy_(baseline_state["fc1.bias"])
        self.fc2.weight.data.copy_(baseline_state["fc2.weight"])
        self.fc2.bias.data.copy_(baseline_state["fc2.bias"])

        for parameter in self.fc1.parameters():
            parameter.requires_grad = False
        for parameter in self.fc2.parameters():
            parameter.requires_grad = False

        self.delta_fc1_weight = nn.Parameter(
            torch.zeros(hidden_size, len(new_feature_indices),
                        dtype=old_weight.dtype)
        )

    def forward(self, x):
        delta_x = x.index_select(1, self.new_feature_indices.to(x.device))
        hidden = F.linear(x, self.fc1.weight, self.fc1.bias)
        hidden = hidden + F.linear(delta_x, self.delta_fc1_weight)
        return torch.sigmoid(self.fc2(torch.relu(hidden)))

    def merged_state_dict(self):
        """Return a normal ANN state_dict representing P' = P + Delta P."""
        state = {
            "fc1.weight": self.fc1.weight.detach().clone(),
            "fc1.bias": self.fc1.bias.detach().clone(),
            "fc2.weight": self.fc2.weight.detach().clone(),
            "fc2.bias": self.fc2.bias.detach().clone(),
        }
        indices = self.new_feature_indices.tolist()
        state["fc1.weight"][:, indices] += self.delta_fc1_weight.detach()
        return state


# ─── ABAC attribute groups ────────────────────────────────────────────────────

designation_grp = [
    ["CEO"], ["CTO"], ["COO"], ["CQO"],
    ["FINANCE MANAGER", "SENIOR_FINANCE_MANAGER"],
    ["HR MANAGER", "SENIOR_HR_MANAGER"],
    ["DESIGNER", "PROGRAMMER", "SDE", "TESTER", "UI_DESIGNER", "UX_DESIGNER"],
    ["PROJECT_MANAGER", "SYSTEM_ARCHITECT"],
    ["PROJECT_LEADER", "PRINCIPAL"],
    ["IT_MANAGER", "SECURTY_ENGINEER"],
    ["NETWORK_ENGINEER"],
    ["DATABASE_ENGINEER", "DATABASE_ARCHITECT"],
    ["QA_LEAD", "TEST_ENGINEER", "AUTOMATION_ENGINEER"],
    ["CUSTOMER_SUCCESS_MANAGER", "SUPPORT_LEAD", "TECHNICAL_SUPPORT_ENGINEER"],
    ["OPS_MANAGER", "DEVOPS_ENGINEER", "SITE_RELIABILITY_ENGINEER"],
]
resource_grp = [
    ["EMP_DETAIL"], ["CLIENT_DETAIL"],
    ["BENEFITS_DETAIL", "USER_DETAIL"],
    ["SALARY_DETAIL", "PF_DETAIL"],
    ["PROJECT_DETAIL", "PROJECT_PLAN", "EMP_DETAIL", "SPRINT_DETAIL"],
    ["NETWORK_SETUP"],
    ["DATABASE", "BACKUP_DATABASE"],
    ["PROJECT_COST", "ALLOCATED_FUND", "FINANCE_REPORT",
     "TAX_DETAIL", "BUDGET_DETAIL"],
    ["SERVER", "STORAGE", "GPU", "CLOUD_SERVER"],
    ["SECURITY_SETUP"],
    ["TEST_DETAIL", "BUG_REPORT", "QA_METRICS", "TEST_COVERAGE"],
    ["TICKET_DETAIL", "CUSTOMER_FEEDBACK", "SUPPORT_METRICS"],
    ["OPS_METRICS", "DEPLOYMENT_DETAIL", "INCIDENT_REPORT"],
]
attr_grp = {"DESIGNATION": designation_grp, "Resource": resource_grp}


# ─── Distributed setup / teardown ────────────────────────────────────────────

def setup_multithread(rank, world_size):
    store = dist.TCPStore(
        "127.0.0.1", MT_PORT, world_size, rank == 0,
        timedelta(seconds=300), use_libuv=False,
    )
    dist.init_process_group(backend="gloo", store=store,
                            rank=rank, world_size=world_size)


def setup_multimachine(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=600),
    )


def cleanup():
    dist.destroy_process_group()


# ─── Weight aggregation ───────────────────────────────────────────────────────

def aggregate_multithread(model, world_size):
    """FedAvg only Delta P; frozen baseline parameters are not reduced."""
    dist.all_reduce(model.delta_fc1_weight.data, op=dist.ReduceOp.SUM)
    model.delta_fc1_weight.data /= world_size


def aggregate_multimachine_delta(rank, model, n_clients=N_CLIENTS):
    """FedAvg only Delta P for the server-plus-client multimachine layout."""
    delta = model.delta_fc1_weight.data
    if rank == 0:
        delta.zero_()
    dist.reduce(delta, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        delta /= n_clients
    dist.broadcast(delta, src=0)


# ─── Attribute mapping helpers ────────────────────────────────────────────────

def _column_keys(columns):
    """Give duplicate CSV columns stable identities (name, occurrence)."""
    counts = {}
    keys = []
    for column in columns:
        occurrence = counts.get(column, 0)
        keys.append((column, occurrence))
        counts[column] = occurrence + 1
    return keys


def GetAttributeMapping(data, grp=None, grp_gap=20, map_type=1):
    """Build an encoding per attribute, never one flat mapping for all columns."""
    mappings = {}
    keys = _column_keys(data.columns)
    grouped_values = {}
    if map_type == 2 and grp:
        for group_name, groups in grp.items():
            grouped_values[group_name] = {
                value: group_number * grp_gap + member_number
                for group_number, members in enumerate(groups, 1)
                for member_number, value in enumerate(members, 1)
            }

    for index, (key, column) in enumerate(zip(keys, data.columns)):
        column_map = {"NotA": -1, 0: 0, "YES": 1, "NO": 0}
        if map_type == 2 and column in grouped_values:
            column_map.update(grouped_values[column])

        # Encode every categorical attribute. This also covers an inserted
        # feature, while preserving the original order-based values for the
        # existing ungrouped columns.
        next_value = 1
        for value in data.iloc[:, index].dropna().unique():
            if value not in column_map:
                column_map[value] = next_value
                next_value += 1
        mappings[key] = column_map
    return mappings


def GetBaselineAttributeMapping(data, grp=None, grp_gap=20, map_type=1):
    """Reproduce the flat mapping used by the original PBP_Company.py."""
    mapping = {"NotA": -1, 0: 0, "YES": 1, "NO": 0}
    if map_type == 1:
        for col in data.columns[:4]:
            count = 1
            for value in data[col].unique():
                if value != "NotA":
                    mapping[value] = count
                    count += 1
    elif map_type == 2:
        for col in data.columns[1:3]:
            count = 1
            for value in data[col].unique():
                if value != "NotA":
                    mapping[value] = count
                    count += 1
        for group_name in grp or {}:
            group_number = 1
            for members in grp[group_name]:
                member_number = 1
                for value in members:
                    mapping[value] = group_number * grp_gap + member_number
                    member_number += 1
                group_number += 1
    return mapping


def encode_baseline_with_new_features(data, baseline_mapping,
                                      new_feature_names=NEW_FEATURE_NAMES):
    """Apply baseline mapping globally and new mappings only to new columns."""
    # Object dtype is required because the targeted new column may change
    # from strings to integer codes after the global baseline replacement.
    encoded = data.replace(baseline_mapping).astype(object)
    new_mapping = dict(baseline_mapping)

    for index, column in enumerate(data.columns):
        if column not in new_feature_names:
            continue

        next_value = 1
        for value in data.iloc[:, index].unique():
            if value != "NotA" and value not in new_mapping:
                new_mapping[value] = next_value
                next_value += 1

        # Use raw values from the original column. This prevents newly added
        # category keys from affecting any existing feature column.
        encoded.iloc[:, index] = data.iloc[:, index].map(
            lambda value: new_mapping.get(value, value)
        )

    return encoded


def encode_attributes(data, mapping, uniform_mapping=False,
                      new_feature_names=None):
    """Apply per-column mappings using duplicate-column-aware identities."""
    if uniform_mapping:
        # This is intentionally the same operation used by the baseline PBP
        # implementation. Existing feature values must retain its assignments.
        if new_feature_names:
            return encode_baseline_with_new_features(
                data, mapping, new_feature_names
            )
        return data.replace(mapping)
    # Object dtype is required because categorical string columns are being
    # replaced by numeric codes.
    encoded = data.copy().astype(object)
    for index, key in enumerate(_column_keys(encoded.columns)):
        column_map = mapping.get(key, {})
        if column_map:
            encoded.iloc[:, index] = encoded.iloc[:, index].replace(column_map)
    return encoded


def same_conditions(col1, col2):
    if col1 == -1 or col2 == -1: return 2
    return 1 if col1 == col2 else 0


def chk_nota(col):
    return 1 if col == -1 else 0


# ─── Data preparation (policy-enforcement artefacts removed) ──────────────────
# Prepare the data for Training and Testing based on relation
def GetPreparedData(train_data, test_data, prep_type=5, seed=0,
                    prep_name="NaiveNA", mapping=None,
                    uniform_mapping=False, new_feature_names=None):
    data = pd.concat([train_data, test_data], axis=0)
    if mapping is None:
        mapping = GetAttributeMapping(data, grp=attr_grp,
                                      map_type=1 if prep_type in (1, 2, 5) else 2)
    if prep_type == 1:  # Naive (Normal encoding)
        map_type = 1
        data_encoded = encode_attributes(
            data, mapping, uniform_mapping, new_feature_names
        )
    elif (
        prep_type == 2
    ):  # Columns for same attribute values in subject and object (ARFE)
        map_type = 1
        data_encoded = encode_attributes(
            data, mapping, uniform_mapping, new_feature_names
        )
        data_encoded["sameProj"] = data_encoded.apply(
            lambda x: same_conditions(x["Project_name"], x["Project_Name"]), axis=1
        )
        data_encoded["sameDep"] = data_encoded.apply(
            lambda x: same_conditions(x["Department"], x["Department.1"]), axis=1
        )
        data_encoded = data_encoded.drop("Department", axis=1)
        data_encoded = data_encoded.drop("Department.1", axis=1)
        data_encoded = data_encoded.drop("Project_name", axis=1)
        data_encoded = data_encoded.drop("Project_Name", axis=1)
    elif (
        prep_type == 3
    ):  # Grouping of attributes (Encoding based on atrribute group) (AVC)
        map_type = 2
        data_encoded = encode_attributes(
            data, mapping, uniform_mapping, new_feature_names
        )
    elif (
        prep_type == 4
    ):  # Grouping of attributes + Columns for same attribute values in subject and object (ARFE + AVC)
        map_type = 2
        data_encoded = encode_attributes(
            data, mapping, uniform_mapping, new_feature_names
        )
        data_encoded["sameProj"] = data_encoded.apply(
            lambda x: same_conditions(x["Project_name"], x["Project_Name"]), axis=1
        )
        data_encoded["sameDep"] = data_encoded.apply(
            lambda x: same_conditions(x["Department"], x["Department.1"]), axis=1
        )
        data_encoded = data_encoded.drop("Department", axis=1)
        data_encoded = data_encoded.drop("Department.1", axis=1)
        data_encoded = data_encoded.drop("Project_name", axis=1)
        data_encoded = data_encoded.drop("Project_Name", axis=1)
    elif prep_type == 5:  # Naive + NACol (Type 1 with extra encoding for NA_Cols)
        map_type = 1
        data_encoded = encode_attributes(
            data, mapping, uniform_mapping, new_feature_names
        )
        data_encoded["Proj_NA"] = data_encoded.apply(
            lambda x: chk_nota(x["Project_name"]), axis=1
        )

    data_encoded = data_encoded.apply(pd.to_numeric, errors="coerce")

    X = data_encoded.loc[:, data_encoded.columns != "Access"]
    y = data_encoded.loc[:, data_encoded.columns == "Access"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=seed
    )

    return X_train, X_test, y_train, y_test, mapping


# ─── Dataset partitioning: best-feature clustering (PBP) ─────────────────────

def partition_dataset(X_train, y_train, best_feature, best_feature_name, n_clients):
    """
    Groups rows by unique values of best_feature; distributes each
    group proportionally across n_clients.
    """
    partition_details = {
        "best_feature": best_feature_name, "clusters": {}, "clients": {}
    }
    unique_values = np.unique(X_train[:, best_feature])
    clusters      = {v: np.where(X_train[:, best_feature] == v)[0]
                     for v in unique_values}

    X_clients = [[] for _ in range(n_clients)]
    y_clients = [[] for _ in range(n_clients)]
    yes_no_counts       = []
    client_indices_list = [[] for _ in range(n_clients)]

    for value, indices in clusters.items():
        np.random.shuffle(indices)
        partition_size = len(indices) // n_clients
        yc = (y_train[indices] == 1).sum().item()
        nc = (y_train[indices] == 0).sum().item()
        partition_details["clusters"][value] = {
            "size": len(indices), "yes_count": yc,
            "no_count": nc, "client_distribution": {},
        }
        for i in range(n_clients):
            s  = i * partition_size
            e  = (i + 1) * partition_size if i != n_clients - 1 else len(indices)
            ci = indices[s:e]
            if len(ci) == 0:
                continue
            X_clients[i].append(X_train[ci])
            y_clients[i].append(y_train[ci])
            client_indices_list[i].extend(ci.tolist())
            cy = (y_train[ci] == 1).sum().item()
            cn = (y_train[ci] == 0).sum().item()
            partition_details["clusters"][value]["client_distribution"][f"Client {i+1}"] = {
                "yes_count": cy, "no_count": cn, "size": len(ci),
            }

    for i in range(n_clients):
        if not X_clients[i]:
            continue
        X_clients[i] = torch.cat(X_clients[i], dim=0)
        y_clients[i] = torch.cat(y_clients[i], dim=0)
        yc = (y_clients[i] == 1).sum().item()
        nc = (y_clients[i] == 0).sum().item()
        yes_no_counts.append((yc, nc))
        partition_details["clients"][i] = {"yes_count": yc, "no_count": nc}
        print(f"Client {i+1}: Yes={yc}  No={nc}")

    return X_clients, y_clients, yes_no_counts, partition_details, client_indices_list

# modified to retain original indices for sample weights
def get_data_loaders(X_clients, y_clients, batch_size=32):
    data_loaders = []
    for client_number, (X, y) in enumerate(zip(X_clients, y_clients), start=1):
        if len(X) == 0:
            raise ValueError(
                f"Client {client_number} has no samples; cannot create a "
                "federated DataLoader."
            )
        indices = torch.arange(len(X))
        dataset = TensorDataset(X, y, indices)
        data_loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
        )
    return data_loaders


# Balancing regulariser
def train_local(rank, model, loader, device, optimizer=None):
    """Train only Delta P on the expanded labelled dataset."""
    if optimizer is None:
        optimizer = optim.Adam([model.delta_fc1_weight], lr=model.learning_rate)
    start = time.time()
    model.train()

    for epoch in range(model.max_iter):
        epoch_loss = 0.0
        batches = 0
        for X_b, y_b, _ in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred = model(X_b)
            loss = model.criterion(pred, y_b).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"[Client {rank}] Epoch {epoch+1}/{model.max_iter}  "
                  f"Loss={epoch_loss / max(batches, 1):.4f}")

    elapsed = time.time() - start
    print(f"[Client {rank}] Training done in {elapsed:.2f}s")
    return elapsed


def evaluate(model, test_loader, device):
    model.eval()
    total = correct = 0
    all_labels, all_preds, inf_times = [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            t0 = time.time()
            X_b, y_b = X_b.to(device), y_b.to(device)
            out  = model(X_b); inf_times.append(time.time() - t0)
            pred = (out >= 0.5).float()
            total   += y_b.size(0)
            correct += (pred == y_b).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
    accuracy  = 100 * correct / total
    al, ap    = np.array(all_labels), np.array(all_preds)
    precision = precision_score(al, ap)
    recall    = recall_score(al, ap)
    f1        = f1_score(al, ap)
    print(f"[Eval] Acc={accuracy:.4f}%  P={precision:.4f}  "
          f"R={recall:.4f}  F1={f1:.4f}")
    return accuracy, precision, recall, f1, inf_times


# ─── Results / logging helpers ────────────────────────────────────────────────

def save_round_result(model_name, accuracy, precision, recall, f1, results_dir):
    path   = os.path.join(results_dir, "results.csv")
    exists = os.path.isfile(path)
    with open(path, "a") as f:
        if not exists:
            f.write("Model,Accuracy,Precision,Recall,F1Score\n")
        f.write(f"{model_name},{accuracy},{precision},{recall},{f1}\n")


def save_training_time(model_name, rank, t, time_file):
    with open(time_file, "a") as f:
        f.write(f"{model_name},{rank},{t}\n")


def initialize_time_file(time_file):
    if not os.path.isfile(time_file):
        with open(time_file, "w") as f:
            f.write("Model,Rank,TrainingTime\n")


def save_consolidated_result(base_dir, algo, prep_name, model_name,
                              acc, prec, rec, f1,
                              train_time_s, avg_inf_ms,
                              train_records, test_records):
    """Append one row to the shared consolidated_results.csv (12 rows total)."""
    csv_path   = os.path.join(base_dir, "consolidated_results.csv")
    fieldnames = ["Algo", "PrepName", "Model",
                  "Accuracy", "Precision", "Recall", "F1",
                  "TrainTime_s", "AvgInference_ms",
                  "TrainRecords", "TestRecords"]
    write_hdr  = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_hdr:
            writer.writeheader()
        writer.writerow({
            "Algo":            algo,
            "PrepName":        prep_name,
            "Model":           model_name,
            "Accuracy":        round(acc,          4),
            "Precision":       round(prec,         4),
            "Recall":          round(rec,          4),
            "F1":              round(f1,           4),
            "TrainTime_s":     round(train_time_s, 4),
            "AvgInference_ms": round(avg_inf_ms,   6),
            "TrainRecords":    train_records,
            "TestRecords":     test_records,
        })
    print(f"[INFO] Consolidated result appended - {csv_path}")


def log_initial_details(model_name, world_size, X_clients,
                        train_len, test_len, train_ync, test_ync,
                        partition_details, log_file):
    with open(log_file, "a") as f:
        f.write(f"Model: {model_name}\nClients: {world_size}\n"
                f"Train: {train_len}  Test: {test_len}\n"
                f"Test YES={test_ync[0]}  NO={test_ync[1]}\n")
        for i, xc in enumerate(X_clients):
            f.write(f"Client {i+1}: {len(xc)} records  "
                    f"YES={train_ync[i][0]}  NO={train_ync[i][1]}\n")
        f.write(f"\nBest Feature (perception): "
                f"{partition_details.get('best_feature', 'N/A')}\n\n")


def aggregate_metrics(current_folder_path):
    results_dir = os.path.join(current_folder_path, "Final")
    os.makedirs(results_dir, exist_ok=True)
    results = []
    for sf in os.listdir(current_folder_path):
        path = os.path.join(current_folder_path, sf, "results.csv")
        if os.path.isfile(path):
            df   = pd.read_csv(path)
            last = df.tail(1).copy()
            last.insert(0, "Seed", sf.split("_")[-1])
            results.append(last)
    if not results:
        return
    final = (pd.concat(results, ignore_index=True)
               .sort_values("Seed").reset_index(drop=True))
    final.to_csv(os.path.join(results_dir, "final_results.csv"), index=False)
    avgs = final.mean(numeric_only=True)
    with open(os.path.join(results_dir, "average_metrics.txt"), "w") as f:
        for col in avgs.index:
            f.write(f"{col}: {avgs[col]}\n")
    print("Aggregated metrics saved.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTITHREAD  mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_multithread(
    rank, world_size, model_class, baseline_state,
    X_train, X_test, y_train, y_test,
    X_clients, y_clients, yes_no_counts, partition_details,
    results_dir, model_path, time_file, log_file,
    base_dir, prep_name, new_feature_indices,
):
    """Federated P' = P + Delta P training on the expanded dataset only."""
    setup_multithread(rank, world_size)
    device = torch.device("cpu")
    model = model_class(
        baseline_state, X_train.shape[1], new_feature_indices
    ).to(device)
    loaders = get_data_loaders(X_clients, y_clients)
    optimizer = optim.Adam([model.delta_fc1_weight], lr=model.learning_rate)

    if rank == 0:
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=32, shuffle=False
        )
        test_yes_no = (
            (y_test == 1).sum().item(), (y_test == 0).sum().item()
        )
        log_initial_details(
            "ArtificialNeuralNetwork", world_size, X_clients,
            len(X_train), len(X_test), yes_no_counts, test_yes_no,
            partition_details, log_file,
        )

    fed_rounds = 50
    model_name = "ArtificialNeuralNetwork"
    start = time.time()
    total_inf_t = []
    acc = prec = rec = f1 = 0.0

    for r in range(fed_rounds):
        if rank == 0:
            print(f"[MT] Round {r+1}/{fed_rounds}")
        train_time = train_local(rank, model, loaders[rank], device, optimizer)
        aggregate_multithread(model, world_size)

        if rank == 0:
            acc, prec, rec, f1, inf_t = evaluate(model, test_loader, device)
            total_inf_t.extend(inf_t)
            save_round_result(model_name, acc, prec, rec, f1, results_dir)
            with open(log_file, "a") as lf:
                lf.write(
                    f"Round {r+1}  TrainTime={train_time:.4f}s  "
                    f"AvgInference={np.mean(inf_t):.10f}s\n"
                )

    total_train_time = time.time() - start
    save_training_time("ArtificialNeuralNetwork", rank, total_train_time, time_file)

    if rank == 0:
        os.makedirs(model_path, exist_ok=True)
        save_path = os.path.join(
            model_path, f"{model_name}_feat_incr_final.pth"
        )
        torch.save(model.merged_state_dict(), save_path)
        print(f"[INFO] Feat. Incr. Model saved -> {save_path}")
        avg_inf_ms = np.mean(total_inf_t) * 1000 if total_inf_t else 0.0
        save_consolidated_result(
            base_dir, ALGO, prep_name, model_name,
            acc, prec, rec, f1, total_train_time, avg_inf_ms,
            int(len(X_train)), int(len(X_test)),
        )

    cleanup()


# New feature-incremental entry point: expanded data + baseline checkpoint.
def main(prep_type):
    prep_names = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}
    prep_name = prep_names.get(prep_type, f"Prep{prep_type}")
    seed = 0
    base_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(
        base_dir, "Results", "feat-incr", "Company", ALGO,
        prep_name, f"Seed_{seed}"
    )
    model_path = os.path.join(base_dir, MODEL_ROOT, prep_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    time_file = os.path.join(results_dir, "train_times.csv")
    log_file = os.path.join(results_dir, "training_log.txt")
    initialize_time_file(time_file)

    # Load the old files only to reproduce the baseline encoding and validate
    # the old processed schema. They are not used as training examples.
    old_train_data = pd.read_csv(os.path.join(base_dir, "Dataset", "train_company.csv"))
    old_test_data = pd.read_csv(os.path.join(base_dir, "Dataset", "test_company.csv"))
    train_data = pd.read_csv(os.path.join(base_dir, "Dataset", "new_train_company.csv"))
    test_data = pd.read_csv(os.path.join(base_dir, "Dataset", "new_test_company.csv"))
    mapping_type = 1 if prep_type in (1, 2, 5) else 2
    baseline_mapping = GetBaselineAttributeMapping(
        pd.concat([old_train_data, old_test_data], axis=0),
        grp=attr_grp, map_type=mapping_type,
    )
    X_old, _, _, _, _ = GetPreparedData(
        old_train_data, old_test_data, prep_type=prep_type, seed=seed,
        prep_name=prep_name, mapping=baseline_mapping,
        uniform_mapping=True,
    )
    X_train, X_test, y_train, y_test, _ = GetPreparedData(
        train_data, test_data, prep_type=prep_type, seed=seed,
        prep_name=prep_name, mapping=baseline_mapping,
        uniform_mapping=True,
        new_feature_names=NEW_FEATURE_NAMES,
    )

    checkpoint_path = os.path.join(
        base_dir, "Model", "old", "PBP", prep_name,
        "ArtificialNeuralNetwork_final.pth",
    )
    baseline_state = load_baseline_state(checkpoint_path)
    old_input_dim = baseline_state["fc1.weight"].shape[1]
    old_feature_indices, new_feature_indices = resolve_checkpoint_feature_indices(
        X_train.columns, X_old.columns, old_input_dim
    )
    print(
        f"[INFO] Baseline={checkpoint_path} | old_features={old_input_dim} | "
        f"expanded_features={X_train.shape[1]} | "
        f"new_positions={new_feature_indices}"
    )

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    np.random.seed(seed)
    best_index, best_name = resolve_best_feature(X_train.columns)
    X_clients, y_clients, yes_no_counts, partition_details, _ = partition_dataset(
        X_train_t, y_train_t, best_index, best_name, N_CLIENTS
    )

    torch.multiprocessing.spawn(
        run_federated_multithread,
        args=(
            N_CLIENTS, ArtificialNeuralNetwork, baseline_state,
            X_train_t, X_test_t, y_train_t, y_test_t,
            X_clients, y_clients, yes_no_counts, partition_details,
            results_dir, model_path, time_file, log_file,
            base_dir, prep_name, new_feature_indices,
        ),
        nprocs=N_CLIENTS, join=True, start_method="spawn",
    )
    aggregate_metrics(os.path.dirname(results_dir))


# ═══════════════════════════════════════════════════════════════════════════════
#  PREP  mode
# ═══════════════════════════════════════════════════════════════════════════════
# New multimachine preparation: persist only the expanded dataset.
def run_prep(prep_type):
    prep_names = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}
    prep_name = prep_names.get(prep_type, f"Prep{prep_type}")
    seed = 0
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_dir = os.path.join(base_dir, "Tensors", ALGO, prep_name)
    os.makedirs(tensor_dir, exist_ok=True)

    old_train = pd.read_csv(os.path.join(base_dir, "Dataset", "train_company.csv"))
    old_test = pd.read_csv(os.path.join(base_dir, "Dataset", "test_company.csv"))
    train = pd.read_csv(os.path.join(base_dir, "Dataset", "new_train_company.csv"))
    test = pd.read_csv(os.path.join(base_dir, "Dataset", "new_test_company.csv"))
    mapping_type = 1 if prep_type in (1, 2, 5) else 2
    baseline_mapping = GetBaselineAttributeMapping(
        pd.concat([old_train, old_test]), grp=attr_grp, map_type=mapping_type
    )
    X_old, _, _, _, _ = GetPreparedData(
        old_train, old_test, prep_type=prep_type, seed=seed,
        prep_name=prep_name, mapping=baseline_mapping, uniform_mapping=True,
    )
    X_train, X_test, y_train, y_test, _ = GetPreparedData(
        train, test, prep_type=prep_type, seed=seed,
        prep_name=prep_name, mapping=baseline_mapping,
        uniform_mapping=True, new_feature_names=NEW_FEATURE_NAMES,
    )
    checkpoint_path = os.path.join(
        base_dir, "Model", "old", "PBP", prep_name,
        "ArtificialNeuralNetwork_final.pth",
    )
    baseline_state = load_baseline_state(checkpoint_path)
    old_input_dim = baseline_state["fc1.weight"].shape[1]
    old_indices, new_indices = resolve_checkpoint_feature_indices(
        X_train.columns, X_old.columns, old_input_dim
    )

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    np.random.seed(seed)
    best_index, best_name = resolve_best_feature(X_train.columns)
    clients = partition_dataset(
        X_train_t, y_train_t, best_index, best_name, N_CLIENTS
    )
    X_clients, y_clients = clients[0], clients[1]
    for i in range(N_CLIENTS):
        torch.save(X_clients[i], os.path.join(tensor_dir, f"client_{i+1}_X.pt"))
        torch.save(y_clients[i], os.path.join(tensor_dir, f"client_{i+1}_y.pt"))
    torch.save(X_test_t, os.path.join(tensor_dir, "test_X.pt"))
    torch.save(y_test_t, os.path.join(tensor_dir, "test_y.pt"))
    torch.save({
        "old_feature_indices": old_indices,
        "new_feature_indices": new_indices,
        "input_dim": X_train.shape[1],
    }, os.path.join(tensor_dir, "feature_indices.pt"))
    print(f"[PREP] Expanded features: {X_train.shape[1]}")
    print(f"[PREP] Tensors saved -> {tensor_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTIMACHINE  mode
# ═══════════════════════════════════════════════════════════════════════════════
def run_federated_multimachine(rank, world_size, prep_type, prep_name,
                               master_addr, master_port, base_dir):
    """Run the same P' = P + Delta P method across physical machines."""
    setup_multimachine(rank, world_size, master_addr, master_port)
    device = torch.device("cpu")
    tensor_dir = os.path.join(base_dir, "Tensors", ALGO, prep_name)
    results_dir = os.path.join(
        base_dir, "Results", "feat-incr", "Company", ALGO,
        prep_name, "Seed_0"
    )
    model_path = os.path.join(base_dir, MODEL_ROOT, prep_name)
    log_file = os.path.join(results_dir, "training_log_mm.txt")
    time_file = os.path.join(results_dir, "train_times_mm.csv")

    feature_indices = torch.load(
        os.path.join(tensor_dir, "feature_indices.pt"), map_location="cpu"
    )
    new_feature_indices = feature_indices["new_feature_indices"]
    checkpoint_path = os.path.join(
        base_dir, "Model", "old", "PBP", prep_name,
        "ArtificialNeuralNetwork_final.pth",
    )
    baseline_state = load_baseline_state(checkpoint_path)

    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        initialize_time_file(time_file)
        X_test = torch.load(os.path.join(tensor_dir, "test_X.pt"), map_location="cpu")
        y_test = torch.load(os.path.join(tensor_dir, "test_y.pt"), map_location="cpu")
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), batch_size=32, shuffle=False
        )
        input_dim = X_test.shape[1]
        n_test = len(X_test)
        print(f"[Server rank=0] Expanded features: {input_dim} | test={n_test}")
    else:
        X_client = torch.load(
            os.path.join(tensor_dir, f"client_{rank}_X.pt"), map_location="cpu"
        )
        y_client = torch.load(
            os.path.join(tensor_dir, f"client_{rank}_y.pt"), map_location="cpu"
        )
        client_loader = DataLoader(
            TensorDataset(X_client, y_client, torch.arange(len(X_client))),
            batch_size=32, shuffle=True,
        )
        input_dim = X_client.shape[1]
        print(f"[Client rank={rank}] Expanded records: {len(X_client)}")

    model = ArtificialNeuralNetwork(
        baseline_state, input_dim, new_feature_indices
    ).to(device)
    optimizer = optim.Adam([model.delta_fc1_weight], lr=model.learning_rate)
    fed_rounds = 50
    start = time.time()
    total_inf_t = []
    acc = prec = rec = f1 = 0.0

    for round_number in range(fed_rounds):
        if rank == 0:
            print(f"[MM] Round {round_number + 1}/{fed_rounds}")
        else:
            train_local(rank, model, client_loader, device, optimizer)

        aggregate_multimachine_delta(rank, model, n_clients=N_CLIENTS)

        if rank == 0:
            acc, prec, rec, f1, inf_t = evaluate(model, test_loader, device)
            total_inf_t.extend(inf_t)
            save_round_result(
                "ArtificialNeuralNetwork", acc, prec, rec, f1, results_dir
            )
            with open(log_file, "a") as log:
                log.write(
                    f"Round {round_number + 1}  "
                    f"AvgInference={np.mean(inf_t):.10f}s\n"
                )

    total_train_time = time.time() - start
    save_training_time("ArtificialNeuralNetwork", rank, total_train_time, time_file)
    if rank == 0:
        save_path = os.path.join(
            model_path, "ArtificialNeuralNetwork_feat_incr_final.pth"
        )
        torch.save(model.merged_state_dict(), save_path)
        print(f"[Server] Model saved -> {save_path}")
        avg_inf_ms = np.mean(total_inf_t) * 1000 if total_inf_t else 0.0
        save_consolidated_result(
            base_dir, ALGO, prep_name, "ArtificialNeuralNetwork",
            acc, prec, rec, f1, total_train_time, avg_inf_ms,
            -1, int(n_test),
        )
    cleanup()


def main_multimachine(prep_type, rank, master_addr, master_port):
    prep_names = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}
    prep_name  = prep_names.get(prep_type, f"Prep{prep_type}")
    base_dir   = os.path.dirname(os.path.abspath(__file__))

    print(f"\n[{ALGO}] prep_type={prep_type} ({prep_name})  "
          f"rank={rank}  mode=multimachine  master={master_addr}:{master_port}")
    run_federated_multimachine(rank, WORLD_SIZE_MM, prep_type, prep_name,
                               master_addr, master_port, base_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{ALGO} Federated Learning",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["multithread", "prep", "multimachine"],
        default="multithread",
        help=(
            "multithread  : simulate 4 clients + server on one machine (default)\n"
            "prep         : prepare & save tensors/CSVs for multimachine mode\n"
            "multimachine : real distributed - one process per VM\n"
        ),
    )
    parser.add_argument(
        "--prep_type", type=int, choices=[2, 3, 4, 5], default=None,
        help="Encoding type (omit to run all 4 in multithread/prep modes)",
    )
    parser.add_argument("--rank", type=int, default=0,
                        help="Process rank  [multimachine]  0=server, 1-4=clients")
    parser.add_argument("--master_addr", default="127.0.0.1",
                        help="Server VM IP  [multimachine only]")
    parser.add_argument("--master_port", type=int, default=29502,
                        help="Server VM port  [multimachine only]")
    args = parser.parse_args()

    if args.mode == "multithread":
        prep_types = [args.prep_type] if args.prep_type else [2, 3, 4, 5]
        for pt in prep_types:
            main(pt)
            print(f"Completed prep_type={pt}")

    elif args.mode == "prep":
        prep_types = [args.prep_type] if args.prep_type else [2, 3, 4, 5]
        for pt in prep_types:
            run_prep(pt)
            print(f"[PREP] Done prep_type={pt}")

    else:
        if args.prep_type is None:
            parser.error("--prep_type is required for multimachine mode")
        main_multimachine(args.prep_type, args.rank,
                          args.master_addr, args.master_port)

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


def resolve_best_feature(columns):
    """Return (index, name) of the perception feature within the encoded X."""
    cols = list(columns)
    if BEST_FEATURE_NAME in cols:
        return cols.index(BEST_FEATURE_NAME), BEST_FEATURE_NAME
    # Fallback: first column (keeps the run alive if the name was dropped)
    return 0, cols[0]


# ─── Model definitions ────────────────────────────────────────────────────────

class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layer_size=64,
                 learning_rate=0.001, max_iter=200):
        super().__init__()
        self.input_dim         = input_dim
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate     = learning_rate
        self.max_iter          = max_iter
        self.fc1               = nn.Linear(input_dim, hidden_layer_size)
        self.fc2               = nn.Linear(hidden_layer_size, 1)
        self.relu              = nn.ReLU()
        self.sigmoid           = nn.Sigmoid()
        self.criterion         = nn.BCELoss(reduction=None) # per sample bce loss is needed
        self.optimizer         = optim.Adam

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


models_dict = {"ArtificialNeuralNetwork": ArtificialNeuralNetwork}


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
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size


def aggregate_multimachine(model, rank, n_clients=N_CLIENTS):
    """
    FedAvg: reduce to server (rank 0) → average over n_clients → broadcast.
    Server acts as pure aggregator; its initialisation weights are not
    included in the average (divide by n_clients, not world_size).
    """
    for param in model.parameters():
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            param.data /= n_clients
        dist.broadcast(param.data, src=0)


# ─── Attribute mapping helpers ────────────────────────────────────────────────

def GetAttributeMapping(data, grp=None, grp_gap=20, map_type=1):
    mapping = {"NotA": -1, 0: 0, "YES": 1, "NO": 0}
    if map_type == 1:
        for col in data.columns[:4]:
            cnt = 1
            for val in data[col].unique():
                if val != "NotA":
                    mapping[val] = cnt; cnt += 1
    elif map_type == 2:
        for col in data.columns[1:3]:
            cnt = 1
            for val in data[col].unique():
                if val != "NotA":
                    mapping[val] = cnt; cnt += 1
        for g in grp:
            grp_num = 1
            for member in grp[g]:
                mem_num = 1
                for val in member:
                    mapping[val] = grp_num * grp_gap + mem_num; mem_num += 1
                grp_num += 1
    return mapping


def same_conditions(col1, col2):
    if col1 == -1 or col2 == -1: return 2
    return 1 if col1 == col2 else 0


def chk_nota(col):
    return 1 if col == -1 else 0


# ─── Data preparation (policy-enforcement artefacts removed) ──────────────────
# Prepare the data for Training and Testing based on relation
def GetPreparedData(train_data, test_data, prep_type=5, seed=0,prep_name="NaiveNA"):
    data = pd.concat([train_data, test_data], axis=0)
    if prep_type == 1:  # Naive (Normal encoding)
        map_type = 1
        mapping = GetAttributeMapping(data, grp=attr_grp, map_type=map_type)
        print(mapping)
        data_encoded = data.replace(mapping)
    elif (
        prep_type == 2
    ):  # Columns for same attribute values in subject and object (ARFE)
        map_type = 1
        mapping = GetAttributeMapping(data, grp=attr_grp, map_type=map_type)
        data_encoded = data.replace(mapping)
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
        mapping = GetAttributeMapping(data, grp=attr_grp, map_type=map_type)
        data_encoded = data.replace(mapping)
    elif (
        prep_type == 4
    ):  # Grouping of attributes + Columns for same attribute values in subject and object (ARFE + AVC)
        map_type = 2
        mapping = GetAttributeMapping(data, grp=attr_grp, map_type=map_type)
        data_encoded = data.replace(mapping)
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
        mapping = GetAttributeMapping(data, grp=attr_grp, map_type=map_type)
        data_encoded = data.replace(mapping)
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
    for X, y in zip(X_clients, y_clients):
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
def calculating_balancing_regulariser(X, u):
    """
    Calculate the balancing regularizer Phi_u(X)
    from Eq. (7) of Ni et al. -> Phi_u(X) = || X~ (T-C) ||^2

    X: [n_samples, n_features]
    u: [n_samples]
    """

    n_samples, n_features = X.shape

    total_bal = torch.tensor(0.0, device=X.device, dtype=X.dtype)

    ones = torch.ones(n_samples, device=X.device, dtype=X.dtype)

    for j in range(n_features):
        # Treatment feature X_j
        X_j = X[:, j]

        # remaining features X_(-j)
        remaining_mask = torch.ones(n_features, device=X.device, dtype=torch.bool)
        remaining_mask[j] = False

        X_remaining = X[:, remaining_mask]

        # weighted treatment/control denominators
        # T_j = u ⊙ X_j / u^T X_j
        # C_j = u ⊙ (1 - X_j) / u^T (1 - X_j)

        T_wt = u @ X_j
        C_wt = u @ (ones - X_j)
        eps = 1e-8 # avoid div by 0

        T_j = u * X_j / (T_wt + eps)
        C_j = u * (ones - X_j) / (C_wt + eps)

        # weighted first-order moment difference
        T_M = X_remaining.T @ T_j
        C_M = X_remaining.T @ C_j
        M_diff = T_M - C_M

        total_bal += torch.sum(M_diff ** 2)

    return total_bal

def model_wt_regulariser(model):
    return sum(
        torch.sum(param ** 2) for param in model.parameters()
    )

# ─── Train / Test ─────────────────────────────────────────────────────────────

def train_local(rank, prev_model, prev_loader, curr_model, curr_loader, device):

    n_old = len(prev_loader.dataset)
    n_curr = len(curr_loader.dataset)

    # Paper parameterizes non-negative sample weights as:
    # u = v ⊙ v
    # this guarantees u >= 0
    # Initialize u_i = 1 / n so that sum(u) = 1 initially.
    v1 = nn.Parameter(
        torch.full(
            (n_old,),
            1.0 / math.sqrt(n_old),
            device=device
        )
    )

    v2 = nn.Parameter(
        torch.full(
            (n_curr,),
            1.0 / math.sqrt(n_curr),
            device=device
        )
    )

    # Jointly optimize:
    #   1. ANN parameters
    #   2. sample-weight parameters v
    
    # Separate optimizers for the four alternating updates
    opt_w1 = optim.Adam(
        prev_model.parameters(),
        lr=prev_model.learning_rate
    )

    opt_w2 = optim.Adam(
        curr_model.parameters(),
        lr=curr_model.learning_rate
    )

    opt_u1 = optim.Adam(
        [v1],
        lr=prev_model.learning_rate
    )

    opt_u2 = optim.Adam(
        [v2],
        lr=curr_model.learning_rate
    )

    start = time.time()
    prev_model.train()
    curr_model.train()

    for epoch in range(prev_model.max_iter):
        
        ### Update w(2). Fix w(1), u(1), u(2)
        u1, u2 = (v1 * v1).detach(), (v2 * v2).detach()

        for X_b, y_b, i_b in curr_loader:
            X_b, y_b, i_b = X_b.to(device), y_b.to(device), i_b.to(device)
            u2_b = u2[i_b]

            pred = curr_model(X_b)
            bce = curr_model.criterion(pred, y_b)
            pred_loss = torch.sum(u2_b * bce)

            # w(1) fixed here
            consistency_reg = LAMBDA_CON * torch.sum(
                (prev_model.fc1.weight.detach() - curr_model.fc1.weight[:, :-1]) ** 2
            )

            model_reg = LAMBDA_W * model_wt_regulariser(curr_model)

            loss = pred_loss + consistency_reg + model_reg

            opt_w2.zero_grad(); loss.backward(); opt_w2.step()

        ### Update w(1). Fix w(2), u(1), u(2)
        u1, u2 = (v1 * v1).detach(), (v2 * v2).detach()

        for X_b, y_b, i_b in prev_loader:
            X_b, y_b, i_b = X_b.to(device), y_b.to(device), i_b.to(device)
            u1_b = u1[i_b]

            pred = prev_model(X_b)
            bce = prev_model.criterion(pred, y_b)

            pred_loss = torch.sum(u1_b * bce)
            # w(2) fixed here
            consistency_reg = LAMBDA_CON * torch.sum(
                (prev_model.fc1.weight - curr_model.fc1.weight[:, :-1].detach()) ** 2
            )
            model_reg = LAMBDA_W * model_wt_regulariser(prev_model)
            loss = pred_loss + consistency_reg + model_reg

            opt_w1.zero_grad(); loss.backward(); opt_w1.step()

        ### Update u(1) through v(1). Fix w(1), w(2), u(2)
        for X_b, y_b, i_b in prev_loader:
            X_b, y_b, i_b = X_b.to(device), y_b.to(device), i_b.to(device)
            u1 = v1 * v1
            u1_b = u1[i_b]

            with torch.no_grad(): # to fix weights
                pred = prev_model(X_b)
                bce = prev_model.criterion(pred, y_b)

            pred_loss = torch.sum(u1_b * bce)
            bal_reg = LAMBDA_BAL * calculating_balancing_regulariser(X=X_b, u=u1_b)
            sample_wt_reg = LAMBDA_U * torch.sum(u1 ** 2)
            sample_wt_norm = LAMBDA_U_NORM * (torch.sum(u1) - 1.0) ** 2
            loss = pred_loss + bal_reg + sample_wt_reg + sample_wt_norm

            opt_u1.zero_grad(); loss.backward(); opt_u1.step()

        ### Update u(2) through v(2). Fix w(1), w(2), u(1)
        for X_b, y_b, i_b in curr_loader:
            X_b, y_b, i_b = X_b.to(device), y_b.to(device), i_b.to(device)
            u2 = v2 * v2
            u2_b = u2[i_b]

            with torch.no_grad(): # to fix weights
                pred = curr_model(X_b)
                bce = curr_model.criterion(pred, y_b)

            pred_loss = torch.sum(u2_b * bce)
            bal_reg = LAMBDA_BAL * calculating_balancing_regulariser(X=X_b, u=u2_b)
            sample_wt_reg = LAMBDA_U * torch.sum(u2 ** 2)
            sample_wt_norm = LAMBDA_U_NORM * (torch.sum(u2) - 1.0) ** 2
            loss = pred_loss + bal_reg + sample_wt_reg + sample_wt_norm

            opt_u2.zero_grad(); loss.backward(); opt_u2.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"[Client {rank}] Epoch {epoch+1}/{prev_model.max_iter}  "
                  f"Loss={loss.item():.4f}")
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
    print(f"[INFO] Consolidated result appended → {csv_path}")


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
    rank, world_size, model_class,
    X_train, X_test, y_train, y_test,
    X_clients, y_clients, yes_no_counts,
    partition_details,
    results_dir, model_path, time_file, log_file,
    base_dir, prep_name,
):
    setup_multithread(rank, world_size)
    device = torch.device("cpu")
    prev_model = model_class(X_train.shape[1]).to(device)
    curr_model = model_class(X_train.shape[1] + 1).to(device)

    # prev_loaders = get_data_loaders(
    #     X_prev_clients,
    #     y_prev_clients
    # )

    # curr_loaders = get_data_loaders(
    #     X_curr_clients,
    #     y_curr_clients
    # )

    if rank == 0:
        test_loader = DataLoader(TensorDataset(X_test, y_test),
                                 batch_size=32, shuffle=False)
        test_ync = ((y_test == 1).sum().item(), (y_test == 0).sum().item())
        log_initial_details(
            model_class.__name__, world_size, X_clients,
            len(X_train), len(X_test),
            yes_no_counts, test_ync,
            partition_details, log_file,
        )

    fed_rounds  = 50
    start       = time.time()
    total_inf_t = []
    acc = prec = rec = f1 = 0.0

    for r in range(fed_rounds):
        if rank == 0:
            print(f"[MT] Round {r+1}/{fed_rounds}")
        train_time = train_local(rank, model, data_loaders[rank], device)
        aggregate_multithread(model, world_size)
        end = time.time()
        if rank == 0:
            acc, prec, rec, f1, inf_t = evaluate(model, test_loader, device)
            total_inf_t.extend(inf_t)
            save_round_result(model_class.__name__, acc, prec, rec, f1, results_dir)
            with open(log_file, "a") as lf:
                lf.write(f"Round {r+1}  TrainTime={train_time:.4f}s  "
                         f"AvgInference={np.mean(inf_t):.10f}s\n")

    total_train_time = end - start
    save_training_time(model_class.__name__, rank, total_train_time, time_file)

    if rank == 0:
        os.makedirs(model_path, exist_ok=True)
        save_path = os.path.join(model_path, f"{model_class.__name__}_final.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Model saved → {save_path}")

        avg_inf_ms = np.mean(total_inf_t) * 1000 if total_inf_t else 0.0
        save_consolidated_result(
            base_dir, ALGO, prep_name, model_class.__name__,
            acc, prec, rec, f1,
            total_train_time, avg_inf_ms,
            int(len(X_train)), int(len(X_test)),
        )

    cleanup()


def main(prep_type):
    prep_names = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}
    prep_name  = prep_names.get(prep_type, f"Prep{prep_type}")

    for seed in range(1):
        print(f"\n{'='*60}")
        print(f"[{ALGO}] prep_type={prep_type} ({prep_name})  "
              f"seed={seed}  mode=multithread")
        print(f"{'='*60}")
        np.random.seed(seed)

        base_dir = current_folder_path = os.path.dirname(os.path.abspath(__file__))

        results_dir     = os.path.join(current_folder_path, "Results", "Company",
                                       ALGO, prep_name, f"Seed_{seed}")
        model_path      = os.path.join(current_folder_path, "Model", ALGO, prep_name)
        os.makedirs(results_dir,     exist_ok=True)
        os.makedirs(model_path,      exist_ok=True)

        time_file = os.path.join(results_dir, "train_times.csv")
        log_file  = os.path.join(results_dir, "training_log.txt")
        initialize_time_file(time_file)

        # Old feature set
        train_data_prev = pd.read_csv(
            os.path.join(current_folder_path, "Dataset", "train_company.csv")
        )
        test_data_prev = pd.read_csv(
            os.path.join(current_folder_path, "Dataset", "test_company.csv")
        )

        X_prev_train, X_prev_test, y_prev_train, y_prev_test, mapping_prev = (
            GetPreparedData(
                train_data_prev, test_data_prev, prep_type=prep_type,
                seed=seed, prep_name=prep_name,
            )
        )

        # Current feature set = old features + one new feature
        train_data_curr = pd.read_csv(
            os.path.join(current_folder_path, "Dataset", "new_train_company.csv")
        )
        test_data_curr = pd.read_csv(
            os.path.join(current_folder_path, "Dataset", "new_test_company.csv")
        )

        X_curr_train, X_curr_test, y_curr_train, y_curr_test, mapping_curr = (
            GetPreparedData(
                train_data_curr, test_data_curr, prep_type=prep_type,
                seed=seed, prep_name=prep_name,
            )
        )
        
        # Resolve the same PBP perception feature in both feature sets
        best_feature_index_prev, best_feature_name_prev = resolve_best_feature(X_prev_train.columns)
        best_feature_index_curr, best_feature_name_curr = resolve_best_feature(X_curr_train.columns)

        # Convert old feature set to tensors
        X_prev_train_t = torch.tensor(X_prev_train.values, dtype=torch.float32)
        X_prev_test_t  = torch.tensor(X_prev_test.values,  dtype=torch.float32)
        y_prev_train_t = torch.tensor(y_prev_train.values, dtype=torch.float32).view(-1, 1)
        y_prev_test_t  = torch.tensor(y_prev_test.values, dtype=torch.float32).view(-1, 1)

        # Convert current feature set to tensors
        X_curr_train_t = torch.tensor(X_curr_train.values, dtype=torch.float32)
        X_curr_test_t  = torch.tensor(X_curr_test.values,  dtype=torch.float32)
        y_curr_train_t = torch.tensor(y_curr_train.values, dtype=torch.float32).view(-1, 1)
        y_curr_test_t  = torch.tensor(y_curr_test.values, dtype=torch.float32).view(-1, 1)

        world_size = N_CLIENTS

        # Partition old feature set
        X_prev_clients, y_prev_clients, prev_yes_no_counts, prev_partition_details, prev_client_indices = partition_dataset(
            X_prev_train_t, y_prev_train_t, best_feature_index_prev, best_feature_name_prev, world_size,
        )

        # Partition current feature set
        X_curr_clients, y_curr_clients, curr_yes_no_counts, curr_partition_details, curr_client_indices = partition_dataset(
            X_curr_train_t, y_curr_train_t, best_feature_index_curr, best_feature_name_curr, world_size,
        )

        for name, model_cls in models_dict.items():
            print(f"[INFO] Training model: {name}")
            torch.multiprocessing.spawn(
                run_federated_multithread,
                args=(
                    world_size, model_cls,
                    X_train_t, X_test_t, y_train_t, y_test_t,
                    X_clients, y_clients, yes_no_counts,
                    partition_details,
                    results_dir, model_path, time_file, log_file,
                    base_dir, prep_name,
                ),
                nprocs=world_size, join=True, start_method="spawn",
            )

    results_parent = os.path.split(results_dir)[0]
    aggregate_metrics(results_parent)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREP  mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_prep(prep_type):
    """Prepare and persist encoded tensors for multimachine training."""
    prep_names = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}
    prep_name  = prep_names.get(prep_type, f"Prep{prep_type}")
    seed       = 0

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    tensor_dir = os.path.join(base_dir, "Tensors", ALGO, prep_name)
    os.makedirs(tensor_dir, exist_ok=True)

    train_data = pd.read_csv(os.path.join(base_dir, "Dataset", "train_company.csv"))
    test_data  = pd.read_csv(os.path.join(base_dir, "Dataset", "test_company.csv"))

    np.random.seed(seed)
    (X_train, X_test, y_train, y_test, mapping) = GetPreparedData(
        train_data, test_data,
        prep_type=prep_type, seed=seed, prep_name=prep_name,
    )

    # Perception-based feature (fixed by domain knowledge, resolved by name)
    best_feature_index, best_feature_name = resolve_best_feature(X_train.columns)
    print(f"[PREP] Perception feature: '{best_feature_name}' "
          f"(column index {best_feature_index})")

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)

    (X_clients, y_clients, _, _,
     _) = partition_dataset(
        X_train_t, y_train_t, best_feature_index, best_feature_name, N_CLIENTS
    )

    for i in range(N_CLIENTS):
        torch.save(X_clients[i], os.path.join(tensor_dir, f"client_{i+1}_X.pt"))
        torch.save(y_clients[i], os.path.join(tensor_dir, f"client_{i+1}_y.pt"))
    torch.save(X_test_t, os.path.join(tensor_dir, "test_X.pt"))
    torch.save(y_test_t, os.path.join(tensor_dir, "test_y.pt"))

    print(f"[PREP] Tensors saved → {tensor_dir}")
    print(f"[PREP] Ready for: python {__file__} --mode multimachine "
          f"--prep_type {prep_type} --rank <R> --master_addr <IP>")


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTIMACHINE  mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_federated_multimachine(rank, world_size, prep_type, prep_name,
                               master_addr, master_port, base_dir):
    """
    Each physical VM runs this with its assigned rank.
      rank 0  → Global server: aggregates, evaluates, saves model.
      rank 1-4 → FL clients: train locally, participate in FedAvg.
    """
    setup_multimachine(rank, world_size, master_addr, master_port)
    device = torch.device("cpu")

    tensor_dir  = os.path.join(base_dir, "Tensors", ALGO, prep_name)
    results_dir = os.path.join(base_dir, "Results", "Company",
                               ALGO, prep_name, "Seed_0")
    model_path  = os.path.join(base_dir, "Model", ALGO, prep_name)
    log_file    = os.path.join(results_dir, "training_log_mm.txt")
    time_file   = os.path.join(results_dir, "train_times_mm.csv")

    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(model_path,  exist_ok=True)
        initialize_time_file(time_file)

    if rank == 0:
        X_test_t = torch.load(os.path.join(tensor_dir, "test_X.pt"))
        y_test_t = torch.load(os.path.join(tensor_dir, "test_y.pt"))
        input_dim   = X_test_t.shape[1]
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t),
                                 batch_size=32, shuffle=False)
        n_test = len(X_test_t)
        print(f"[Server rank=0] Test set: {n_test} records | input_dim={input_dim}")
    else:
        X_client = torch.load(os.path.join(tensor_dir, f"client_{rank}_X.pt"))
        y_client = torch.load(os.path.join(tensor_dir, f"client_{rank}_y.pt"))
        input_dim    = X_client.shape[1]
        local_loader = DataLoader(TensorDataset(X_client, y_client),
                                  batch_size=32, shuffle=True)
        print(f"[Client rank={rank}] Local data: {len(X_client)} records")

    model = ArtificialNeuralNetwork(input_dim).to(device)

    fed_rounds  = 50
    start       = time.time()
    total_inf_t = []
    acc = prec = rec = f1 = 0.0

    for r in range(fed_rounds):
        if rank == 0:
            print(f"[Server] ── Aggregation round {r+1}/{fed_rounds} ──")
        else:
            print(f"[Client {rank}] Training round {r+1}/{fed_rounds}")

        if rank != 0:
            train_local(rank, model, local_loader, device)

        aggregate_multimachine(model, rank, n_clients=N_CLIENTS)
        end = time.time()

        if rank == 0:
            acc, prec, rec, f1, inf_t = evaluate(model, test_loader, device)
            total_inf_t.extend(inf_t)
            save_round_result(model.__class__.__name__,
                              acc, prec, rec, f1, results_dir)
            with open(log_file, "a") as lf:
                lf.write(f"Round {r+1}  AvgInference={np.mean(inf_t):.10f}s\n")

    total_train_time = end - start
    save_training_time(model.__class__.__name__, rank, total_train_time, time_file)

    if rank == 0:
        save_path = os.path.join(model_path, f"{model.__class__.__name__}_final.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[Server] Model saved → {save_path}")

        avg_inf_ms = np.mean(total_inf_t) * 1000 if total_inf_t else 0.0
        save_consolidated_result(
            base_dir, ALGO, prep_name, model.__class__.__name__,
            acc, prec, rec, f1,
            total_train_time, avg_inf_ms,
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
            "multimachine : real distributed — one process per VM\n"
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

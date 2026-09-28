"""
SBP_U1.py  –  Simple Best-feature Clustering federated learning (University_1)
===============================================================================
Runs all 4 prep_types (ARFE=2, AVC=3, ARFE+AVC=4, NaiveNA=5) in sequence.

University_1 Dataset columns
──────────────────────────────
  Designation, Department, Degree, Year, Type,
  Department.1, Degree.1, Year.1, Access

╔══════════════════════════════════════════════════════════╗
║  MODES                                                   ║
╠══════════════════════════════════════════════════════════╣
║  multithread  (default)                                  ║
║    Simulates 4 client threads + server on ONE machine.   ║
║    Uses torch.multiprocessing.spawn; world_size = 4.     ║
║    Run:  python SBP_U1.py                                ║
║                                                          ║
║  prep                                                    ║
║    Data-preparation step for multimachine mode.          ║
║    Run once on the machine that has the dataset.         ║
║    Saves encoded tensors (.pt) to Tensors/ folder.       ║
║    Run:  python SBP_U1.py --mode prep                    ║
║          python SBP_U1.py --mode prep --prep_type 2      ║
║                                                          ║
║  multimachine                                            ║
║    Real distributed: one process per physical VM.        ║
║    world_size = 5  (rank 0 = global server, 1-4 = FL     ║
║    clients). Communicates over TCP via Gloo backend.     ║
║                                                          ║
║    Prerequisites: run 'prep' mode first.                 ║
║                                                          ║
║    On server VM (rank 0):                                ║
║      python SBP_U1.py --mode multimachine \\             ║
║             --rank 0 --master_addr <SERVER_IP> \\        ║
║             --master_port 29504 --prep_type 2            ║
║    On each client VM (ranks 1-4):                        ║
║      python SBP_U1.py --mode multimachine \\             ║
║             --rank <R> --master_addr <SERVER_IP> \\      ║
║             --master_port 29504 --prep_type 2            ║
╚══════════════════════════════════════════════════════════╝

Artefacts saved
───────────────
  Model/U1/SBC/<prep_name>/ArtificialNeuralNetwork_final.pth
  Results/U1/SBC/<prep_name>/Seed_0/results.csv
  consolidated_results.csv   ← 12-case summary across all algos/prep_types
"""

import argparse
import csv
import os
import random
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ALGO          = "SBC"
DATASET       = "U1"
N_CLIENTS     = 4
WORLD_SIZE_MM = 5       # 1 server (rank 0) + 4 clients (ranks 1-4)
MT_PORT       = 7777    # localhost TCPStore port for multithread mode


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
        self.criterion         = nn.BCELoss()
        self.optimizer         = optim.Adam

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


models_dict = {"ArtificialNeuralNetwork": ArtificialNeuralNetwork}


# ─── ABAC attribute groups ────────────────────────────────────────────────────

designation_grp = [["officer"], ["prof", "adj_prof", "vis_prof"], ["stu"]]
type_grp = [
    ["asgn", "quiz"],
    ["off_rec", "dept_bud", "proj"],
    ["std_mat"],
    ["attdn", "stu_rec"],
    ["q_pr", "grade_book"],
]
department_grp = [
    ["Math", "Phy", "Chy", "Bio"],
    ["Civil", "Electrical"],
    ["Life Science", "Earth Science"],
]
attr_grp = {
    "Designation": designation_grp,
    "Type":        type_grp,
    "Department":  department_grp,
}


# ─── Distributed setup / teardown ─────────────────────────────────────────────

def setup_multithread(rank, world_size):
    store = dist.TCPStore(
        "localhost", MT_PORT, world_size, rank == 0,
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
    FedAvg for world_size = n_clients + 1 (server = rank 0).
    reduce → server averages by n_clients → broadcast.
    """
    for param in model.parameters():
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            param.data /= n_clients
        dist.broadcast(param.data, src=0)


# ─── Attribute mapping helpers ────────────────────────────────────────────────

def GetAttributeMapping(data, grp=None, grp_gap=20, map_type=1):
    mapping = {"NotA": -1, 0: 0, "Yes": 1, "No": 0}
    if map_type == 1:
        for col in data.columns[:8]:
            cnt = 1
            for val in data[col].unique():
                if val != "NotA":
                    mapping[val] = cnt; cnt += 1
    elif map_type == 2:
        for col in data.columns[2:6]:
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

def GetPreparedData(train_data, test_data, prep_type=5, seed=0, k_best=1,
                    prep_name="NaiveNA"):
    """
    Encodes U1 data and finds best feature via SelectKBest.
    REMOVED: scaler .pkl, mapping .pkl, raw TestData CSV saves.
    Returns: X_train, X_test, y_train, y_test,
             best_feature_index, best_feature_ordering, mapping
    """
    data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    # Temporary encoding for feature selection
    data_temp = data.copy()
    for col in data_temp.columns:
        if data_temp[col].dtype == "object":
            le = LabelEncoder()
            data_temp[col] = le.fit_transform(data_temp[col])

    X_temp = data_temp.loc[:, data_temp.columns != "Access"]
    y_temp = data_temp.loc[:, data_temp.columns == "Access"]

    scaler        = StandardScaler()
    X_temp_scaled = scaler.fit_transform(X_temp)

    selector = SelectKBest(score_func=f_classif, k=k_best)
    selector.fit(X_temp_scaled, y_temp)
    feature_ordering      = sorted(
        zip(X_temp.columns, selector.scores_),
        key=lambda x: x[1], reverse=True,
    )
    best_feature_ordering = [f for f, _ in feature_ordering]
    print("Feature ordering (best to worst):", best_feature_ordering)

    # ── Encoding per prep_type ────────────────────────────────────────────────
    if prep_type == 2:        # ARFE
        mapping      = GetAttributeMapping(data, grp=attr_grp, map_type=1)
        data_encoded = data.replace(mapping)
        data_encoded["sameDep"] = data_encoded.apply(
            lambda x: same_conditions(x["Department"], x["Department.1"]), axis=1)
        data_encoded["sameDeg"] = data_encoded.apply(
            lambda x: same_conditions(x["Degree"], x["Degree.1"]), axis=1)
        data_encoded["sameYr"]  = data_encoded.apply(
            lambda x: same_conditions(x["Year"], x["Year.1"]), axis=1)
        data_encoded = data_encoded.drop(
            ["Department", "Department.1", "Degree", "Degree.1", "Year", "Year.1"], axis=1)

    elif prep_type == 3:      # AVC
        mapping      = GetAttributeMapping(data, grp=attr_grp, map_type=2)
        data_encoded = data.replace(mapping)

    elif prep_type == 4:      # ARFE + AVC
        mapping      = GetAttributeMapping(data, grp=attr_grp, map_type=2)
        data_encoded = data.replace(mapping)
        data_encoded["sameDep"] = data_encoded.apply(
            lambda x: same_conditions(x["Department"], x["Department.1"]), axis=1)
        data_encoded["sameDeg"] = data_encoded.apply(
            lambda x: same_conditions(x["Degree"], x["Degree.1"]), axis=1)
        data_encoded["sameYr"]  = data_encoded.apply(
            lambda x: same_conditions(x["Year"], x["Year.1"]), axis=1)
        data_encoded = data_encoded.drop(
            ["Department", "Department.1", "Degree", "Degree.1", "Year", "Year.1"], axis=1)

    elif prep_type == 5:      # NaiveNA
        mapping      = GetAttributeMapping(data, grp=attr_grp, map_type=1)
        data_encoded = data.replace(mapping)
        data_encoded["Year_NA"]     = data_encoded["Year"].apply(chk_nota)
        data_encoded["Year.1_NA"]   = data_encoded["Year.1"].apply(chk_nota)
        data_encoded["Degree_NA"]   = data_encoded["Degree"].apply(chk_nota)
        data_encoded["Degree.1_NA"] = data_encoded["Degree.1"].apply(chk_nota)

    else:
        mapping      = GetAttributeMapping(data, grp=attr_grp, map_type=1)
        data_encoded = data.replace(mapping)

    # Resolve best feature index in encoded columns
    best_feature_index = -1
    for feat in best_feature_ordering:
        if feat in data_encoded.columns:
            best_feature_index = list(data_encoded.columns).index(feat)
            break

    X = data_encoded.loc[:, data_encoded.columns != "Access"]
    y = data_encoded.loc[:, data_encoded.columns == "Access"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test, best_feature_index, best_feature_ordering, mapping


# ─── Dataset partitioning: best-feature clustering ───────────────────────────

def partition_dataset(X_train, y_train, best_feature, best_feature_name, n_clients):
    partition_details = {
        "best_feature": best_feature_name, "clusters": {}, "clients": {}
    }
    unique_values = np.unique(X_train[:, best_feature])
    clusters      = {v: np.where(X_train[:, best_feature] == v)[0]
                     for v in unique_values}

    X_clients = [[] for _ in range(n_clients)]
    y_clients = [[] for _ in range(n_clients)]
    yes_no_counts = []

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
            if len(ci) == 0: continue
            X_clients[i].append(X_train[ci])
            y_clients[i].append(y_train[ci])
            cy = (y_train[ci] == 1).sum().item()
            cn = (y_train[ci] == 0).sum().item()
            partition_details["clusters"][value]["client_distribution"][f"Client {i+1}"] = {
                "yes_count": cy, "no_count": cn, "size": len(ci),
            }

    for i in range(n_clients):
        if not X_clients[i]: continue
        X_clients[i] = torch.cat(X_clients[i], dim=0)
        y_clients[i] = torch.cat(y_clients[i], dim=0)
        yc = (y_clients[i] == 1).sum().item()
        nc = (y_clients[i] == 0).sum().item()
        yes_no_counts.append((yc, nc))
        partition_details["clients"][i] = {"yes_count": yc, "no_count": nc}
        print(f"Client {i+1}: Yes={yc}  No={nc}")

    return X_clients, y_clients, yes_no_counts, partition_details


def get_data_loaders(X_clients, y_clients, batch_size=32, rank=0, seed=0):
    loaders = []
    for i, (X, y) in enumerate(zip(X_clients, y_clients)):
        g = torch.Generator()
        g.manual_seed(seed * 10 + rank + i)
        loaders.append(DataLoader(TensorDataset(X, y), batch_size=batch_size,
                                  shuffle=True, generator=g))
    return loaders


# ─── Train / Test ─────────────────────────────────────────────────────────────

def train_local(rank, model, train_loader, device):
    optimizer = model.optimizer(model.parameters(), lr=model.learning_rate)
    start = time.time()
    model.train()
    for epoch in range(model.max_iter):
        epoch_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            loss = model.criterion(model(X_b), y_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"[Client {rank}] Epoch {epoch+1}/{model.max_iter}  "
                  f"Loss={epoch_loss/len(train_loader):.4f}")
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
    precision = precision_score(al, ap, zero_division=0)
    recall    = recall_score(al, ap, zero_division=0)
    f1        = f1_score(al, ap, zero_division=0)
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


def save_consolidated_result(base_dir, algo, dataset, prep_name, model_name,
                              acc, prec, rec, f1,
                              train_time_s, avg_inf_ms,
                              train_records, test_records):
    """Append one row to the shared consolidated_results.csv (12 rows total)."""
    csv_path   = os.path.join(base_dir, "consolidated_results.csv")
    fieldnames = ["Algo", "Dataset", "PrepName", "Model",
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
            "Dataset":         dataset,
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
                        partition_details, best_feature_ordering, log_file):
    with open(log_file, "a") as f:
        f.write(f"Model: {model_name}\nClients: {world_size}\n"
                f"Train: {train_len}  Test: {test_len}\n"
                f"Test YES={test_ync[0]}  NO={test_ync[1]}\n")
        for i, xc in enumerate(X_clients):
            f.write(f"Client {i+1}: {len(xc)} records  "
                    f"YES={train_ync[i][0]}  NO={train_ync[i][1]}\n")
        f.write("\nFeature Ordering:\n")
        for feat in best_feature_ordering:
            f.write(f"  {feat}\n")
        f.write(f"\nBest Feature: {partition_details.get('best_feature', 'N/A')}\n\n")


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
    partition_details, best_feature_ordering,
    results_dir, model_path, time_file, log_file,
    base_dir, prep_name, seed=0,
):
    setup_multithread(rank, world_size)

    _seed = seed * 10 + rank
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device       = torch.device("cpu")
    model        = model_class(X_train.shape[1]).to(device)
    data_loaders = get_data_loaders(X_clients, y_clients, rank=rank, seed=seed)

    if rank == 0:
        test_loader = DataLoader(TensorDataset(X_test, y_test),
                                 batch_size=32, shuffle=False)
        test_ync = ((y_test == 1).sum().item(), (y_test == 0).sum().item())
        log_initial_details(
            model_class.__name__, world_size, X_clients,
            len(X_train), len(X_test),
            yes_no_counts, test_ync,
            partition_details, best_feature_ordering, log_file,
        )

    fed_rounds  = 10
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
            base_dir, ALGO, DATASET, prep_name, model_class.__name__,
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
        print(f"[{ALGO}_{DATASET}] prep_type={prep_type} ({prep_name})  "
              f"seed={seed}  mode=multithread")
        print(f"{'='*60}")
        np.random.seed(seed)

        base_dir = current_folder_path = os.path.dirname(os.path.abspath(__file__))

        results_dir = os.path.join(current_folder_path, "Results", DATASET,
                                   ALGO, prep_name, f"Seed_{seed}")
        model_path  = os.path.join(current_folder_path, "Model", DATASET, ALGO, prep_name)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(model_path,  exist_ok=True)

        time_file = os.path.join(results_dir, "train_times.csv")
        log_file  = os.path.join(results_dir, "training_log.txt")
        initialize_time_file(time_file)

        train_data = pd.read_csv(os.path.join(current_folder_path, "Dataset", "train_U1.csv"))
        test_data  = pd.read_csv(os.path.join(current_folder_path, "Dataset", "test_U1.csv"))

        (X_train, X_test, y_train, y_test,
         best_feature_index, best_feature_ordering, _) = GetPreparedData(
            train_data, test_data,
            prep_type=prep_type, seed=seed, k_best=1, prep_name=prep_name,
        )

        if best_feature_index == -1:
            best_feature_index = 0
            best_feature_name  = "No best feature remained"
        else:
            best_feature_name = X_train.columns[best_feature_index]

        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)

        world_size = N_CLIENTS
        (X_clients, y_clients, yes_no_counts,
         partition_details) = partition_dataset(
            X_train_t, y_train_t, best_feature_index, best_feature_name, world_size
        )

        for name, model_cls in models_dict.items():
            print(f"[INFO] Training model: {name}")
            torch.multiprocessing.spawn(
                run_federated_multithread,
                args=(
                    world_size, model_cls,
                    X_train_t, X_test_t, y_train_t, y_test_t,
                    X_clients, y_clients, yes_no_counts,
                    partition_details, best_feature_ordering,
                    results_dir, model_path, time_file, log_file,
                    base_dir, prep_name, seed,
                ),
                nprocs=world_size, join=True, start_method="spawn",
            )

    results_parent = os.path.split(results_dir)[0]
    aggregate_metrics(results_parent)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREP  mode
# ═══════════════════════════════════════════════════════════════════════════════

def run_prep(prep_type):
    """Prepare and save encoded tensors for multimachine training."""
    prep_names = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}
    prep_name  = prep_names.get(prep_type, f"Prep{prep_type}")
    seed       = 0

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    tensor_dir = os.path.join(base_dir, "Tensors", DATASET, ALGO, prep_name)
    os.makedirs(tensor_dir, exist_ok=True)

    train_data = pd.read_csv(os.path.join(base_dir, "Dataset", "train_U1.csv"))
    test_data  = pd.read_csv(os.path.join(base_dir, "Dataset", "test_U1.csv"))

    np.random.seed(seed)
    (X_train, X_test, y_train, y_test,
     best_feature_index, _, _) = GetPreparedData(
        train_data, test_data,
        prep_type=prep_type, seed=seed, k_best=1, prep_name=prep_name,
    )

    if best_feature_index == -1:
        best_feature_index = 0
        best_feature_name  = "No best feature remained"
    else:
        best_feature_name = X_train.columns[best_feature_index]

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)

    (X_clients, y_clients, _, _) = partition_dataset(
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
    rank 0  → Global server: aggregates, evaluates, saves model + results.
    rank 1-4 → FL clients: train locally, participate in FedAvg.
    """
    setup_multimachine(rank, world_size, master_addr, master_port)
    device = torch.device("cpu")

    tensor_dir  = os.path.join(base_dir, "Tensors", DATASET, ALGO, prep_name)
    results_dir = os.path.join(base_dir, "Results", DATASET,
                               ALGO, prep_name, "Seed_0")
    model_path  = os.path.join(base_dir, "Model", DATASET, ALGO, prep_name)
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

    fed_rounds  = 10
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
            base_dir, ALGO, DATASET, prep_name, model.__class__.__name__,
            acc, prec, rec, f1,
            total_train_time, avg_inf_ms,
            -1, int(n_test),
        )

    cleanup()


def main_multimachine(prep_type, rank, master_addr, master_port):
    prep_names = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}
    prep_name  = prep_names.get(prep_type, f"Prep{prep_type}")
    base_dir   = os.path.dirname(os.path.abspath(__file__))

    print(f"\n[{ALGO}_{DATASET}] prep_type={prep_type} ({prep_name})  "
          f"rank={rank}  mode=multimachine  master={master_addr}:{master_port}")
    run_federated_multimachine(rank, WORLD_SIZE_MM, prep_type, prep_name,
                               master_addr, master_port, base_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{ALGO}_{DATASET} Federated Learning",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["multithread", "prep", "multimachine"],
        default="multithread",
        help=(
            "multithread  : simulate 4 clients + server on one machine (default)\n"
            "prep         : prepare & save tensors for multimachine mode\n"
            "multimachine : real distributed — one process per VM\n"
        ),
    )
    parser.add_argument(
        "--prep_type", type=int, choices=[2, 3, 4, 5], default=None,
        help="Encoding type (omit to run all 4 in multithread/prep modes)",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Process rank  [multimachine]  0=server, 1-4=clients",
    )
    parser.add_argument(
        "--master_addr", default="127.0.0.1",
        help="Server VM IP  [multimachine only]",
    )
    parser.add_argument(
        "--master_port", type=int, default=29504,
        help="Server VM port  [multimachine only]",
    )
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

    else:   # multimachine
        if args.prep_type is None:
            parser.error("--prep_type is required for multimachine mode")
        main_multimachine(args.prep_type, args.rank,
                          args.master_addr, args.master_port)

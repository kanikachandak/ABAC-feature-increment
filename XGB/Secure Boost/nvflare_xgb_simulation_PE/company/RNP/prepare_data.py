import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # kept in imports in case other code relies on it
from data_preprocessor import GetPreparedData
import torch
import pickle as _pickle

def partition_dataset(X_train, y_train, n_clients):
    indices = np.random.permutation(len(X_train))
    partition_size = len(X_train) // n_clients

    X_clients, y_clients = [], []
    yes_no_counts = []

    for i in range(n_clients):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i != n_clients - 1 else len(X_train)
        client_indices = indices[start_idx:end_idx]

        X_clients.append(X_train[client_indices])
        y_clients.append(y_train[client_indices])

        yes_count = (y_train[client_indices] == 1).sum().item()
        no_count = (y_train[client_indices] == 0).sum().item()
        yes_no_counts.append((yes_count, no_count))
        print(f"Client {i+1}: Yes = {yes_count}, No = {no_count}")

    return X_clients, y_clients, yes_no_counts




def prepare_data_split(prep_type=5, seed=0, n_clients=4, require_valid_placeholder=True, valid_rows=50):
    print(f"--- Starting Custom Data Preparation For Seed {seed} ---")
    np.random.seed(seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    source_train_file = os.path.join(base_dir, "train.csv")
    source_test_file = os.path.join(base_dir, "test.csv")
    processed_test_file = os.path.join(base_dir, f"test_processed.csv")
    target_col = "Access"

    if not all(os.path.exists(f) for f in [source_train_file, source_test_file]):
        print(f"ERROR: Ensure 'train.csv' and 'test.csv' exist in {base_dir}")
        return
        
    train_df_raw = pd.read_csv(source_train_file)
    test_df_raw = pd.read_csv(source_test_file)

    X_train, X_test, y_train, y_test, mapping = GetPreparedData(
        train_df_raw, test_df_raw, prep_type=prep_type, seed=seed
    )

  

    test_processed = pd.concat(
        [y_test.reset_index(drop=True), X_test.reset_index(drop=True)],
        axis=1
    )
    test_processed.to_csv(processed_test_file, index=False)
    print(f"Saved processed test data to: {processed_test_file}")

    # ── Save mapping + feature cols for Policy Enforcement ────────────────────
    PREP_NAMES_XGB = {2: "ARFE", 3: "AVC", 4: "ARFE_AVC", 5: "NaiveNA"}
    _prep_name_xgb = PREP_NAMES_XGB.get(prep_type, f"Prep{prep_type}")
    _mapping_dir_xgb = os.path.join(base_dir, "Mapping")
    os.makedirs(_mapping_dir_xgb, exist_ok=True)
    _mapping_pkl_xgb = os.path.join(_mapping_dir_xgb,
        f"mapping_Company_RNP_{_prep_name_xgb}.pkl")
    with open(_mapping_pkl_xgb, "wb") as _mf:
        _pickle.dump(mapping, _mf)
    print(f"[PE] Mapping saved → {_mapping_pkl_xgb}")

    # ── Save RAW test data for Policy Enforcement test runner ─────────────────
    # The test runner needs original string values (e.g. "CEO", "DATABASE"),
    # NOT encoded integers. We re-index test_df_raw using the same split indices
    # used by GetPreparedData so rows align exactly with y_test/X_test.
    _raw_test_dir = os.path.join(base_dir, "TestData")
    os.makedirs(_raw_test_dir, exist_ok=True)
    _raw_test_path = os.path.join(_raw_test_dir, f"raw_test_data_{seed}.csv")
    # Recover original test indices via the same split (seed, test_size=0.2)
    
    _all_raw = pd.concat([train_df_raw, test_df_raw], axis=0).reset_index(drop=True)
    _all_idx  = np.arange(len(_all_raw))
    _, _test_idx = train_test_split(_all_idx, shuffle=True, test_size=0.2, random_state=seed)
    _raw_test_df  = _all_raw.iloc[_test_idx].reset_index(drop=True)
    # Also attach the encoded columns with ENC_ prefix for reference
    _enc_cols_df  = X_test.reset_index(drop=True).add_prefix("ENC_")
    _raw_combined = pd.concat([_raw_test_df, _enc_cols_df], axis=1)
    _raw_combined.to_csv(_raw_test_path, index=False)
    print(f"[PE] Raw test data saved → {_raw_test_path}")
    _feat_txt_xgb = os.path.join(_mapping_dir_xgb,
        f"feature_cols_Company_RNP_{_prep_name_xgb}.txt")
    with open(_feat_txt_xgb, "w") as _ff:
        _ff.write("\n".join(list(X_train.columns)))
    print(f"[PE] Feature cols saved → {_feat_txt_xgb}")

    feature_cols = list(X_train.columns)
    cols = [target_col] + feature_cols

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)


    

    X_clients, y_clients, _ = partition_dataset(X_train_t, y_train_t, n_clients)

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    for i, (Xi, yi) in enumerate(zip(X_clients, y_clients), start=1):
        site_dir = os.path.join(data_dir, f"site-{i}")
        os.makedirs(site_dir, exist_ok=True)

        df_train = pd.DataFrame(
            np.hstack([yi.numpy().astype(int), Xi.numpy()]),
            columns=cols
        )

        train_out = os.path.join(site_dir, "train.csv")
        df_train.to_csv(train_out, index=False)
        print(f"Saved client {i} train CSV in {site_dir} | rows={len(df_train)}")

        if require_valid_placeholder:
            k = max(1, min(valid_rows, len(df_train)))
            df_valid = df_train.iloc[:k].copy()
            valid_out = os.path.join(site_dir, "valid.csv")
            df_valid.to_csv(valid_out, index=False)
            print(f"Saved placeholder valid.csv (k={k}) for client {i}")

    print("--- Data Preparation Complete ---")


if __name__ == "__main__":
    prepare_data_split()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

designation_grp = [
    ["CEO"],
    ["CTO"],
    ["COO"],
    ["CQO"],
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
    ["EMP_DETAIL"],
    ["CLIENT_DETAIL"],
    ["BENEFITS_DETAIL", "USER_DETAIL"],
    ["SALARY_DETAIL", "PF_DETAIL"],
    ["PROJECT_DETAIL", "PROJECT_PLAN", "EMP_DETAIL", "SPRINT_DETAIL"],
    ["NETWORK_SETUP"],
    ["DATABASE", "BACKUP_DATABASE"],
    ["PROJECT_COST", "ALLOCATED_FUND", "FINANCE_REPORT", "TAX_DETAIL", "BUDGET_DETAIL"],
    ["SERVER", "STORAGE", "GPU", "CLOUD_SERVER"],
    ["SECURITY_SETUP"],
    ["TEST_DETAIL", "BUG_REPORT", "QA_METRICS", "TEST_COVERAGE"],
    ["TICKET_DETAIL", "CUSTOMER_FEEDBACK", "SUPPORT_METRICS"],
    ["OPS_METRICS", "DEPLOYMENT_DETAIL", "INCIDENT_REPORT"],
]
attr_grp = {"DESIGNATION": designation_grp, "Resource": resource_grp}

def GetAttributeMapping(data, grp=None, grp_gap=20, map_type=1):
    mapping = {}
 
    mapping["NotA"] = -1
    mapping[0] = 0
    mapping["YES"] = 1
    mapping["NO"] = 0
 
    if map_type == 1:
        for col in data.columns[:4]:
            col_un = data[col].unique()
            cnt = 1
            for val in col_un:
                if val != "NotA":
                    mapping[val] = cnt
                    cnt = cnt + 1
        return mapping
    elif map_type == 2:
        for col in data.columns[1:3]:
            col_un = data[col].unique()
            cnt = 1
            for val in col_un:
                if val != "NotA":
                    mapping[val] = cnt
                    cnt = cnt + 1
        for g in grp:
            grp_num = 1
            for member in grp[g]:
                mem_num = 1
                for val in member:
                    mapping[val] = grp_num * grp_gap + mem_num
                    mem_num = mem_num + 1
                grp_num = grp_num + 1
        return mapping
 


# Find Relation between common subject and object attributes
def same_conditions(col1, col2):
    if (col1 == -1) or (col2 == -1):
        return 2
    elif col1 == col2:
        return 1
    else:
        return 0
 
 
def chk_nota(col):
    if col == -1:
        return 1
    else:
        return 0


def GetPreparedData(train_data, test_data, prep_type=5, seed=0, k_best=5):
    data = train_data
    # data = pd.concat([train_data, test_data], axis=0)
    label_encoders = {}

    # Temporarily encode text data to allow feature selection
    data_temp_encoded = data.copy()
    for column in data.columns:
        if data_temp_encoded[column].dtype == "object":
            le = LabelEncoder()
            data_temp_encoded[column] = le.fit_transform(data_temp_encoded[column])
            label_encoders[column] = le

    # Select K best features using SelectKBest on temporary encoded data
    X_temp = data_temp_encoded.loc[:, data_temp_encoded.columns != "Access"]
    y_temp = data_temp_encoded.loc[:, data_temp_encoded.columns == "Access"]
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_temp_scaled = scaler.fit_transform(X_temp)

    selector_temp = SelectKBest(score_func=f_classif, k=k_best)
    selector_temp.fit(X_temp_scaled, y_temp)

    # Get feature scores and order them
    feature_scores = selector_temp.scores_
    feature_ordering = sorted(zip(X_temp.columns, feature_scores), key=lambda x: x[1], reverse=True)

    # Create a list of features ordered by their scores
    best_feature_ordering = [feature for feature, score in feature_ordering]
    print("Feature ordering (best to worst):", best_feature_ordering)

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

    # Find the best feature that remains after encoding
    best_feature_index = -1
    for feature in best_feature_ordering:
        if feature in data_encoded.columns:
            best_feature_index = data_encoded.columns.get_loc(feature)
            break

    X = data_encoded.loc[:, data_encoded.columns != "Access"]
    y = data_encoded.loc[:, data_encoded.columns == "Access"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=seed
    )

    return X_train, X_test, y_train, y_test, best_feature_index, best_feature_ordering, mapping

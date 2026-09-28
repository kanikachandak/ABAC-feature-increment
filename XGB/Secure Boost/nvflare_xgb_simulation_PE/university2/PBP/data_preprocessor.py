import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

designation_grp = [
    ["Accountant"],
    ["Faculty"],
    ["Placement_coordinator", "Stud_representative"],
    ["Stud"],
    ["Teaching_assis"],
]
post_grp = [
    ["Associate", "Assistant", "Temporary"],
    ["Graduate", "Undergraduate"],
    ["PhD"],
    ["Non_Teaching"],
]
type_grp = [
    ["assgn", "report_assgn"],
    ["ac_details", "budget", "payment_details"],
    ["mids_paper", "compre_paper", "answer_sheet"],
    ["result"],
    ["attendance"],
    ["placement_details", "grade_book"],
]
attr_grp = {"Designation": designation_grp, "Post": post_grp, "Type": type_grp}

def GetAttributeMapping(data, grp=None, grp_gap=20, map_type=1):
    mapping = {}
    mapping["NotA"] = -1
    mapping[0] = 0
    mapping["Yes"] = 1
    mapping["No"] = 0
    if map_type == 1:
        for col in data.columns[:7]:
            col_un = data[col].unique()
            cnt = 1
            for val in col_un:
                if val != "NotA":
                    mapping[val] = cnt
                    cnt = cnt + 1
        return mapping
    elif map_type == 2:
        for col in data.columns[2:6]:
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


# Prepare the data for Training and Testing based on relation
def GetPreparedData(train_data, test_data, prep_type=5, seed=0):
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
        data_encoded["sameCourse"] = data_encoded.apply(
            lambda x: same_conditions(x["Course"], x["Course.1"]), axis=1
        )
        data_encoded["sameDep"] = data_encoded.apply(
            lambda x: same_conditions(x["Department"], x["Department.1"]), axis=1
        )
        data_encoded["sameDeg"] = data_encoded.apply(
            lambda x: same_conditions(x["Degree"], x["Degree.1"]), axis=1
        )
        data_encoded["sameYr"] = data_encoded.apply(
            lambda x: same_conditions(x["Year"], x["Year.1"]), axis=1
        )
        data_encoded = data_encoded.drop("Department", axis=1)
        data_encoded = data_encoded.drop("Department.1", axis=1)
        data_encoded = data_encoded.drop("Course", axis=1)
        data_encoded = data_encoded.drop("Course.1", axis=1)
        data_encoded = data_encoded.drop("Degree", axis=1)
        data_encoded = data_encoded.drop("Degree.1", axis=1)
        data_encoded = data_encoded.drop("Year", axis=1)
        data_encoded = data_encoded.drop("Year.1", axis=1)
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
        data_encoded["sameCourse"] = data_encoded.apply(
            lambda x: same_conditions(x["Course"], x["Course.1"]), axis=1
        )
        data_encoded["sameDep"] = data_encoded.apply(
            lambda x: same_conditions(x["Department"], x["Department.1"]), axis=1
        )
        data_encoded["sameDeg"] = data_encoded.apply(
            lambda x: same_conditions(x["Degree"], x["Degree.1"]), axis=1
        )
        data_encoded["sameYr"] = data_encoded.apply(
            lambda x: same_conditions(x["Year"], x["Year.1"]), axis=1
        )
        data_encoded = data_encoded.drop("Department", axis=1)
        data_encoded = data_encoded.drop("Department.1", axis=1)
        data_encoded = data_encoded.drop("Course", axis=1)
        data_encoded = data_encoded.drop("Course.1", axis=1)
        data_encoded = data_encoded.drop("Degree", axis=1)
        data_encoded = data_encoded.drop("Degree.1", axis=1)
        data_encoded = data_encoded.drop("Year", axis=1)
        data_encoded = data_encoded.drop("Year.1", axis=1)
    elif prep_type == 5:  # Naive + NACol (Type 1 with extra encoding for NA_Cols)
        map_type = 1
        mapping = GetAttributeMapping(data, grp=attr_grp, map_type=map_type)
        data_encoded = data.replace(mapping)
        data_encoded["Post_NA"] = data_encoded.apply(
            lambda x: chk_nota(x["Post"]), axis=1
        )
        data_encoded["Course_NA"] = data_encoded.apply(
            lambda x: chk_nota(x["Course"]), axis=1
        )
        data_encoded["Degree_NA"] = data_encoded.apply(
            lambda x: chk_nota(x["Degree"]), axis=1
        )
        data_encoded["Year_NA"] = data_encoded.apply(
            lambda x: chk_nota(x["Year"]), axis=1
        )
        data_encoded["Course.1_NA"] = data_encoded.apply(
            lambda x: chk_nota(x["Course.1"]), axis=1
        )
        data_encoded["Year.1_NA"] = data_encoded.apply(
            lambda x: chk_nota(x["Year.1"]), axis=1
        )

    X = data_encoded.loc[:, data_encoded.columns != "Access"]
    y = data_encoded.loc[:, data_encoded.columns == "Access"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.2, random_state=seed
    )

    return X_train, X_test, y_train, y_test, mapping

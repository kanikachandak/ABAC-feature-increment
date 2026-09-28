# dataset_registry.py
# Central registry for all dataset-specific configurations.
# The unified utils.py reads this to handle company / university1 / university2
# without duplicating code.

# ─────────────────────────── COMPANY ────────────────────────────────────────
_company_designation_grp = [
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
_company_resource_grp = [
    ["EMP_DETAIL"], ["CLIENT_DETAIL"],
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

# ─────────────────────────── UNIVERSITY 1 ───────────────────────────────────
_uni1_designation_grp = [
    ["officer"],
    ["prof", "adj_prof", "vis_prof"],
    ["stu"],
]
_uni1_type_grp = [
    ["asgn", "quiz"],
    ["off_rec", "dept_bud", "proj"],
    ["std_mat"],
    ["attdn", "stu_rec"],
    ["q_pr", "grade_book"],
]
_uni1_department_grp = [
    ["Math", "Phy", "Chy", "Bio"],
    ["Civil", "Electrical"],
    ["Life Science", "Earth Science"],
]

# ─────────────────────────── UNIVERSITY 2 ───────────────────────────────────
_uni2_designation_grp = [
    ["Accountant"],
    ["Faculty"],
    ["Placement_coordinator", "Stud_representative"],
    ["Stud"],
    ["Teaching_assis"],
]
_uni2_post_grp = [
    ["Associate", "Assistant", "Temporary"],
    ["Graduate", "Undergraduate"],
    ["PhD"],
    ["Non_Teaching"],
]
_uni2_type_grp = [
    ["assgn", "report_assgn"],
    ["ac_details", "budget", "payment_details"],
    ["mids_paper", "compre_paper", "answer_sheet"],
    ["result"],
    ["attendance"],
    ["placement_details", "grade_book"],
]


# ─────────────────────────── REGISTRY ───────────────────────────────────────
DATASETS = {
    # ── company ─────────────────────────────────────────────────────────────
    "company": {
        "attr_grp": {
            "DESIGNATION": _company_designation_grp,
            "Resource":    _company_resource_grp,
        },
        "mapping_t1_start": 0, "mapping_t1_end": 4,
        "mapping_t2_start": 1, "mapping_t2_end": 3,
        "arfe_pairs": [
            ("Project_name", "Project_Name", "sameProj"),
            ("Department",   "Department.1", "sameDep"),
        ],
        "arfe_drop": ["Project_name", "Project_Name", "Department", "Department.1"],
        "na_cols": [("Project_name", "Proj_NA")],
        "yes_label":  "YES",
        "no_label":   "NO",
        "nota_label": "NotA",
        # FIX: "DESIGNATION" is present in every prep_type (never dropped).
        # Old value "Department" is dropped by ARFE prep_types 2 and 4.
        "pbp_feature":       "DESIGNATION",
        "pbp_fallback_index": 0,
    },

    # ── university1 ─────────────────────────────────────────────────────────
    "university1": {
        "attr_grp": {
            "Designation": _uni1_designation_grp,
            "Type":        _uni1_type_grp,
            "Department":  _uni1_department_grp,
        },
        "mapping_t1_start": 0, "mapping_t1_end": 7,
        "mapping_t2_start": 2, "mapping_t2_end": 6,
        "arfe_pairs": [
            ("Department", "Department.1", "sameDep"),
            ("Degree",     "Degree.1",     "sameDeg"),
            ("Year",       "Year.1",       "sameYr"),
        ],
        "arfe_drop": ["Department", "Department.1", "Degree", "Degree.1", "Year", "Year.1"],
        "na_cols": [
            ("Year",     "Year_NA"),
            ("Year.1",   "Year.1_NA"),
            ("Degree",   "Degree_NA"),
            ("Degree.1", "Degree.1_NA"),
        ],
        "yes_label":  "Yes",
        "no_label":   "No",
        "nota_label": "NotA",
        # "Type" (resource type) survives every prep_type.
        "pbp_feature":       "Type",
        "pbp_fallback_index": 1,
    },

    # ── university2 ─────────────────────────────────────────────────────────
    "university2": {
        "attr_grp": {
            "Designation": _uni2_designation_grp,
            "Post":        _uni2_post_grp,
            "Type":        _uni2_type_grp,
        },
        "mapping_t1_start": 0, "mapping_t1_end": 7,
        "mapping_t2_start": 2, "mapping_t2_end": 6,
        "arfe_pairs": [
            ("Course",     "Course.1",     "sameCourse"),
            ("Department", "Department.1", "sameDep"),
            ("Degree",     "Degree.1",     "sameDeg"),
            ("Year",       "Year.1",       "sameYr"),
        ],
        "arfe_drop": [
            "Course", "Course.1", "Department", "Department.1",
            "Degree", "Degree.1", "Year", "Year.1",
        ],
        "na_cols": [
            ("Post",     "Post_NA"),
            ("Course",   "Course_NA"),
            ("Degree",   "Degree_NA"),
            ("Year",     "Year_NA"),
            ("Course.1", "Course.1_NA"),
            ("Year.1",   "Year.1_NA"),
        ],
        "yes_label":  "Yes",
        "no_label":   "No",
        "nota_label": "NotA",
        # "Type" survives every prep_type.
        "pbp_feature":       "Type",
        "pbp_fallback_index": 2,
    },
}

# Approximate neg/pos class ratios (used as fallback if CSV is unavailable).
# utils.py recomputes the exact value from the actual training CSV.
DATASET_CLASS_RATIOS = {
    "company":     10.3,
    "university1":  7.5,
    "university2": 34.9,
}


def get_dataset_config(dataset_name: str) -> dict:
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid choices: {list(DATASETS.keys())}"
        )
    return DATASETS[dataset_name]


def get_class_ratio(dataset_name: str) -> float:
    """Return the approximate neg/pos class ratio for scale_pos_weight."""
    return DATASET_CLASS_RATIOS.get(dataset_name, 1.0)

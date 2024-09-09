if True:
    from reset_random import reset_random

    reset_random()
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tqdm

UNSW_NB15_CLASSES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
]
CAR_HACKING_CLASSES = ["DOS", "FUZZY", "GEAR", "RPM"]


def replace_categorical_cols(df, class_col):
    cat_cols = list(set(df.columns) - set(df._get_numeric_data().columns))
    for col in cat_cols:
        if col != class_col:
            print("[INFO] Replacing Categorical Values in Column :: {0}".format(col))
            repd = {v: k + 1 for k, v in enumerate(sorted(df[col].unique()))}
            df[col] = df[col].map(repd)
    df.reset_index(drop=True, inplace=True)
    return df


def load_unswnb15():
    train_path = "Data/source/UNSW_NB15/UNSW_NB15_training-set.csv"
    print("[INFO] Reading Data From :: {0}".format(train_path))
    train_df = pd.read_csv(train_path)
    print("[INFO] Data Shape :: {0}".format(train_df.shape))
    test_path = "Data/source/UNSW_NB15/UNSW_NB15_testing-set.csv"
    print("[INFO] Reading Data From :: {0}".format(test_path))
    test_df = pd.read_csv(test_path)
    print("[INFO] Data Shape :: {0}".format(test_df.shape))
    print("[INFO] Merging Data")
    df = pd.concat([train_df, test_df])
    print("[INFO] Data Shape :: {0}".format(df.shape))
    return df


def preprocess_unsw_nb15(df: pd.DataFrame):
    if df.isna().sum().sum():
        print("[INFO] Dropping Null Values")
        df.dropna(inplace=True)
    ucols = ["id", "label"]
    print("[INFO] Dropping Unwanted Columns :: {0}".format(ucols))
    df.drop(ucols, inplace=True, axis=1)
    df = replace_categorical_cols(df, "attack_cat")
    df = pd.concat([df[df["attack_cat"] == c].head(1500) for c in UNSW_NB15_CLASSES])
    print("[INFO] Replacing Attack Category To Numerical")
    repd = {v: 0 if v == "Normal" else 1 for v in UNSW_NB15_CLASSES}
    df["attack_cat"].replace(repd, inplace=True)
    print("[INFO] Scaling Data Using Standard Scaler")
    # ss = StandardScaler()
    # x = ss.fit_transform(df.values[:, :-1])
    x = df.values[:, :-1]
    y = df.values[:, -1]
    # df = pd.DataFrame(x, columns=list(df.columns)[:-1])
    # df["Class"] = y
    sd = "Data/preprocessed"
    os.makedirs(sd, exist_ok=True)
    sp = os.path.join(sd, "UNSW_NB15.csv")
    print("[INFO] Preprocessed Data Shape :: {0}".format(df.shape))
    print("[INFO] Saving PreProcessed Data :: {0}".format(sp))
    df.to_csv(sp, index=False)
    return df, x, y


def load_car_hacking_data():
    files = ["DoS", "Fuzzy", "RPM", "gear"]
    columns = [
        "Timestamp",
        "CAN ID",
        "DLC",
        "DATA[0]",
        "DATA[1]",
        "DATA[2]",
        "DATA[3]",
        "DATA[4]",
        "DATA[5]",
        "DATA[6]",
        "DATA[7]",
        "Flag",
    ]
    dfs = []
    for file in files:
        print("[INFO] Loading Data From :: {0}".format(file))
        df = pd.read_csv(
            "Data/source/CAR_HACKING/{0}_dataset.csv".format(file), names=columns
        )
        df = df[df["Flag"] == "T"]
        df["Flag"] = file.upper()
        df.dropna(inplace=True)
        df = df.head(3 * 60 * 300)
        dfs.append(df)
    df = pd.concat(dfs)
    print("[INFO] Data Shape :: {0}".format(df.shape))
    return df


def creating_timeseries_data(df):
    print("[INFO] Creating Time Series Data")
    LEN = len(df)
    STEP = len(df) // 4
    new_x = []
    new_y = []
    for i in range(0, LEN, STEP):
        x = df.values[i : i + STEP, :-1]
        y = df.values[i : i + STEP, -1][0]
        x = x.reshape((STEP // 30, 8, 30))
        new_x.extend(x)
        new_y.extend([y] * (STEP // 30))
    x = np.array(new_x)
    y = np.array(new_y, dtype=int)
    print("[INFO] Data Shape :: {0}".format(x.shape))
    return df, x, y


def preprocess_car_hacking(df):
    print("[INFO] Preprocessing")
    ucols = ["Timestamp", "CAN ID", "DLC"]
    print("[INFO] Dropping Unwanted Columns :: {0}".format(ucols))
    df = df.drop(ucols, axis=1)
    bytes_ = sorted(df["DATA[0]"].unique().ravel().tolist())
    bytes_to_int = {v: k for k, v in enumerate(bytes_)}
    for col in tqdm.tqdm(
        list(df.columns)[:-1], desc="[INFO] Converting Byte Data To Numeric :"
    ):
        df[col].replace(bytes_to_int, inplace=True)
    print("[INFO] Scaling Data Using Standard Scaler")
    ss = StandardScaler()
    x = ss.fit_transform(df.values[:, :-1])
    y = df.values[:, -1]
    df = pd.DataFrame(x, columns=list(df.columns)[:-1])
    df["Class"] = y
    repd = {v: i for i, v in enumerate(CAR_HACKING_CLASSES)}
    df["Class"].replace(repd, inplace=True)
    return creating_timeseries_data(df)


def load_data(name):
    print("[INFO] Loading {0} Dataset".format(name))
    if name == "UNSW_NB15":
        return load_unswnb15()
    else:
        return load_car_hacking_data()

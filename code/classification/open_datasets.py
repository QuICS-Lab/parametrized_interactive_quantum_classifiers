import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


def load_pima_diabetes(
    replace_invalid_zeros=False,
    scale=False,
):
    """Load the Pima Indians Diabetes dataset."""

    dataset = fetch_openml(
        data_id=37,
        as_frame=True,
        parser="auto",
    )

    X = dataset.data.copy()
    y = dataset.target.copy()

    # OpenML usually stores the target as tested_negative/tested_positive.
    y = y.map({
        "tested_negative": 0,
        "tested_positive": 1,
    })

    # Support versions where the target is already numeric.
    if y.isna().any():
        y = pd.to_numeric(dataset.target, errors="raise").astype(int)

    X = X.apply(pd.to_numeric, errors="raise")

    # These zero values are not clinically plausible and are treated as missing.
    if replace_invalid_zeros:
        invalid_zero_columns = [
            "plas",   # glucose
            "pres",   # blood pressure
            "skin",   # skin thickness
            "insu",   # insulin
            "mass",   # BMI
        ]

        existing_columns = [
            col for col in invalid_zero_columns
            if col in X.columns
        ]

        X[existing_columns] = X[existing_columns].replace(0, np.nan)

        # For rigorous validation, perform median imputation within a Pipeline.
        X[existing_columns] = X[existing_columns].fillna(
            X[existing_columns].median()
        )

    feature_names = X.columns.tolist()

    X = X.to_numpy(dtype=np.float64)
    y = y.to_numpy(dtype=np.int64)

    if scale:
        X = StandardScaler().fit_transform(X)

    return X, y, feature_names


import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.io import arff


def load_caesarian_section():
    """Load the Caesarian Section Classification dataset from UCI."""

    url = (
        "https://archive.ics.uci.edu/static/public/472/"
        "caesarian%2Bsection%2Bclassification%2Bdataset.zip"
    )

    # Download the archive into memory.
    with urllib.request.urlopen(url) as response:
        zip_content = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:

        arff_filename = next(
            filename
            for filename in zip_file.namelist()
            if filename.lower().endswith(".arff")
        )

        print("Found ARFF file:", arff_filename)
        with zip_file.open(arff_filename, mode="r") as binary_file:
            with io.TextIOWrapper(
                binary_file,
                encoding="utf-8",
                errors="replace",
            ) as text_file:
                data, metadata = arff.loadarff(text_file)

    df = pd.DataFrame(data)

    # Decode columns that may still contain bytes.
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].apply(
                lambda value: (
                    value.decode("utf-8")
                    if isinstance(value, bytes)
                    else value
                )
            )

    df = df.apply(pd.to_numeric, errors="raise")

    print("Original columns:", df.columns.tolist())

    if df.shape[1] != 6:
        raise ValueError(
            f"Expected 6 columns, but found {df.shape[1]}: "
            f"{df.columns.tolist()}"
        )

    df.columns = [
        "age",
        "delivery_number",
        "delivery_time",
        "blood_pressure",
        "heart_problem",
        "caesarian",
    ]

    feature_names = [
        "age",
        "delivery_number",
        "delivery_time",
        "blood_pressure",
        "heart_problem",
    ]

    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["caesarian"].to_numpy(dtype=np.int64)

    return X, y, feature_names
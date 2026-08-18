import numpy
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def load_pima_diabetes(
    replace_invalid_zeros=False,
    scale=False,
):
    """
    Carrega a Pima Indians Diabetes Dataset.

    Retorna
    -------
    X : np.ndarray
        Matriz de atributos com shape (768, 8).
    y : np.ndarray
        Vetor binário:
            0 = sem diabetes
            1 = diabetes
    feature_names : list[str]
        Nomes dos atributos.
    """

    dataset = fetch_openml(
        data_id=37,
        as_frame=True,
        parser="auto",
    )

    X = dataset.data.copy()
    y = dataset.target.copy()

    # A classe no OpenML costuma aparecer como:
    # tested_negative / tested_positive
    y = y.map({
        "tested_negative": 0,
        "tested_positive": 1,
    })

    # Segurança para versões em que o alvo já venha numérico
    if y.isna().any():
        y = pd.to_numeric(dataset.target, errors="raise").astype(int)

    X = X.apply(pd.to_numeric, errors="raise")

    # Em algumas análises da Pima, zeros nestas colunas são tratados
    # como valores ausentes, pois não são valores clínicos plausíveis.
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

        # Imputação pela mediana calculada em toda a base.
        # Em validação rigorosa, prefira realizar isso dentro de Pipeline.
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
    """
    Carrega a Caesarian Section Classification Dataset diretamente da UCI.

    Retorna
    -------
    X : np.ndarray
        Matriz de atributos com shape (80, 5).

    y : np.ndarray
        Vetor de classes binárias.

    feature_names : list[str]
        Nomes dos atributos.
    """

    url = (
        "https://archive.ics.uci.edu/static/public/472/"
        "caesarian%2Bsection%2Bclassification%2Bdataset.zip"
    )

    # Baixa o ZIP para a memória
    with urllib.request.urlopen(url) as response:
        zip_content = response.read()

    # Abre o ZIP
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:

        arff_filename = next(
            filename
            for filename in zip_file.namelist()
            if filename.lower().endswith(".arff")
        )

        print("Arquivo ARFF encontrado:", arff_filename)

        # zip_file.open retorna bytes.
        # TextIOWrapper converte o fluxo binário em texto.
        with zip_file.open(arff_filename, mode="r") as binary_file:
            with io.TextIOWrapper(
                binary_file,
                encoding="utf-8",
                errors="replace",
            ) as text_file:
                data, metadata = arff.loadarff(text_file)

    df = pd.DataFrame(data)

    # Decodifica colunas que eventualmente ainda estejam em bytes
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].apply(
                lambda value: (
                    value.decode("utf-8")
                    if isinstance(value, bytes)
                    else value
                )
            )

    # Converte todas as colunas para numérico
    df = df.apply(pd.to_numeric, errors="raise")

    print("Colunas originais:", df.columns.tolist())

    # A base deve possuir cinco atributos e uma variável-alvo
    if df.shape[1] != 6:
        raise ValueError(
            f"Esperadas 6 colunas, mas foram encontradas {df.shape[1]}: "
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
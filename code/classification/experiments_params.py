# from iqc_multitargets import iqc_de_multi_target, iqc_multidimensional_multi_target
from iqc_zhangetal import iqc_zhangetal, iqc_britoetal
from iqc_multdimensional import iqc_multidimensional
# from iqc_de import iqc_de, next_power_of_two
from sklearn.datasets import load_iris, load_wine, make_blobs, make_circles, make_moons

def generate_models_in_dictionary(number_of_features):
    """Return IQC models and their number of trainable parameters."""

    dicModels = {
                    "iqc_zhangetal": 
                    {
                        'iqc': iqc_zhangetal,
                        'number_of_params': number_of_features + 1  # +1 for bias
                    },
                    
                    "iqc_britoetal":
                    {
                        'iqc': iqc_britoetal,
                        'number_of_params': number_of_features + 1  # +1 for bias
                    },

                    "iqc_alfa":
                    {
                        'iqc': iqc_zhangetal,
                        'number_of_params': number_of_features + 4 + 1  # +1 for bias   
                    },
                    "iqc_multidimensional_2":
                    {
                        'iqc': iqc_multidimensional,
                        'number_of_params': number_of_features * 2 + 4 + 1,
                    },
                    "iqc_multidimensional_4":
                    {
                        'iqc': iqc_multidimensional,
                        'number_of_params': number_of_features * 4 + 4 + 1,
                    },
                    "iqc_multidimensional_8":
                    {
                        'iqc': iqc_multidimensional,
                        'number_of_params': number_of_features * 8 + 4 + 1,
                    },
                    # "iqc_de":
                    # {
                    #     'iqc': iqc_de,
                    #     'number_of_params': next_power_of_two(number_of_features) ** 2 + 5,
                    # },
                    
                    # "iqc_de_multitarget_2":
                    # {
                    #     'iqc': iqc_de_multi_target,
                    #     'number_of_params': next_power_of_two(number_of_features)*next_power_of_two(number_of_features)  + 2*4+ 1 # +1 for bias 
                    # },
                    # "iqc_de_multitarget_3":
                    # {
                    #     'iqc': iqc_de_multi_target,
                    #     'number_of_params': next_power_of_two(number_of_features)*next_power_of_two(number_of_features)  + 3*4+ 1 # +1 for bias 
                    # },
                    # "iqc_multidimensional_2_multitarget_2":
                    # {
                    #     'iqc': iqc_multidimensional_multi_target,
                    #     'number_of_params':  number_of_features*2 + 2*4 + 1  # +1 for
                    # },
                    # "iqc_multidimensional_2_multitarget_3":
                    # {
                    #     'iqc': iqc_multidimensional_multi_target,
                    #     'number_of_params':  number_of_features*2 + 3*4 + 1  # +1 for
                    # },
                    # "iqc_multidimensional_4_multitarget_2":
                    # {
                    #     'iqc': iqc_multidimensional_multi_target,
                    #     'number_of_params': number_of_features*4 + 2*4 + 1  # +1 for
                    # },
                    # "iqc_multidimensional_4_multitarget_3":
                    # {
                    #     'iqc': iqc_multidimensional_multi_target,
                    #     'number_of_params': number_of_features*4 + 3*4 + 1  # +1 for
                    # },
                    # "iqc_multidimensional_8_multitarget_2":
                    # {
                    #     'iqc': iqc_multidimensional_multi_target,
                    #     'number_of_params': number_of_features*8 + 2*4 + 1  # +1 for
                    # },
                    # "iqc_multidimensional_8_multitarget_3":
                    # {
                    #     'iqc': iqc_multidimensional_multi_target,
                    #     'number_of_params': number_of_features*8 + 3*4 + 1  # +1 for
                    # },
                }
    return dicModels

import numpy as np

def load_dataset(filename):
    """Load ``X`` and ``y`` arrays from a Python dataset file."""
    namespace = {}
    with open(filename, "r") as f:
        exec(f.read(), namespace)

    X = np.asarray(namespace["X"], dtype=float)
    y = np.asarray(namespace["y"], dtype=int)

    return X, y

def generate_binary_classification_datasets():
    """Return the binary classification datasets used by the experiments."""
    data_bases_binary_classification = {
        "blobs_2D": make_blobs(n_samples=300, n_features=2, centers=2, cluster_std=1.0, random_state=42),
        "blobs_4D": make_blobs(n_samples=300, n_features=4, centers=2, cluster_std=1.0, random_state=42),
        "blobs_8D": make_blobs(n_samples=300, n_features=8, centers=2, cluster_std=1.0, random_state=42),
        "circle": make_circles(n_samples=1000, noise=0.1, random_state=42),
        "moons": make_moons(n_samples=1000, noise=0.1, random_state=42),
        "circle_no_noise": make_circles(n_samples=1000, random_state=42),
        "moons_no_noise": make_moons(n_samples=1000, random_state=42),
        "xor": load_dataset("datasets/xor.py"),
        "stripes": load_dataset("datasets/stripes.py"),
    }
    return data_bases_binary_classification




from open_datasets import load_pima_diabetes, load_caesarian_section

X_pima, y_pima, features_pima = load_pima_diabetes(
    replace_invalid_zeros=True,
    scale=False,
)

X_caesarian, y_caesarian, features_caesarian = (
    load_caesarian_section()
)

def generate_multiclass_datasets():
    

    data_multiclass_bases = {
        # Real-world datasets
        "iris": (
            load_iris().data,
            load_iris().target,
        ),

        "wine": (
            load_wine().data,
            load_wine().target,
        ),

        "pima_diabetes": (
            X_pima,
            y_pima,
        ),
        "caesarian_section": (
            X_caesarian,
            y_caesarian,
        ),

    }
    return data_multiclass_bases
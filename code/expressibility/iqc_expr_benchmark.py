"""Reproducible, state-based expressibility benchmark for IQC variants.

This benchmark computes one global-state fidelity histogram and one KL(Haar)
value per model. It uses the numerical dynamics from the iqc*.py modules and
supports both single-target and multi-target variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax
from jax import numpy as jnp

def get_U_operator_jax(sigmaQ, sigmaE):
    """
    U = exp(+i * (sigma_Q ⊗ sigma_E))  (Eq. 15/22 from the paper).
    Derived from H_int = -ℏg σ_Q⊗σ_E and U(t) = exp(-iH_int t/ℏ).
    """
    interaction = jnp.kron(sigmaQ, sigmaE)
    return jax.scipy.linalg.expm(1j * interaction)

def get_weighted_sigmaQ_jnp(param, iqcpq=False):
    """
    Build sigma_Q.

    - If iqcpq=False: linear combination of Pauli matrices + identity (Eq. 16).
    - If iqcpq=True: builds an n-level Hermitian matrix with fixed diagonal/off-diagonal.
    """
    if iqcpq:
        n = len(param)
        diagonal = jnp.full(n, 1, dtype=complex)
        diagonal[-1] = -jnp.sum(diagonal[:-1])

        off_diagonal = jnp.full((n, n), 1 + 1j, dtype=complex)
        matrix = jnp.zeros((n, n), dtype=complex)
        jnp.fill_diagonal(matrix, diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = off_diagonal[i, j]
                matrix[j, i] = jnp.conj(off_diagonal[i, j])
        return matrix

    sigmaX = jnp.array([[0, 1], [1, 0]], dtype=complex)
    sigmaY = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
    sigmaZ = jnp.array([[1, 0], [0, -1]], dtype=complex)
    identity = jnp.array([[1, 0], [0, 1]], dtype=complex)

    sigmaQ = (
        param[0] * sigmaX
        + param[1] * sigmaY
        + param[2] * sigmaZ
        + param[3] * identity 
    )
    sigmaq_trace = jnp.trace(sigmaQ)
    eps = jnp.finfo(float).eps
    return jnp.array(sigmaQ) / (sigmaq_trace + eps)
    

BASE_MODEL_NAMES = (
    "iqc_zhangetal",
    "iqc_britoetal",
    "iqc_alfa",
    "iqc_multidimensional_2",
    "iqc_multidimensional_4",
    "iqc_multidimensional_8",
)

MODEL_NAMES = BASE_MODEL_NAMES

MODEL_ALIASES = {
    "IQC": "iqc_zhangetal",
    "IQC_AIL": "iqc_britoetal",
    "IQC_MULTI": "iqc_multidimensional_4",
    "IQC_MULTI_2": "iqc_multidimensional_2",
    "IQC_MULTI_4": "iqc_multidimensional_4",
    "IQC_MULTI_8": "iqc_multidimensional_8",
}

ALPHA_FIXED_MODELS = ("iqc_zhangetal", "iqc_britoetal")

@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared by all models in one benchmark run."""

    n_environment_qubits: int = 2
    n_features: int = 4
    n_pairs: int = 10_000
    n_bins: int = 75
    seed: int = 2026
    randomize_alpha: bool = False
    environment_dimension: int = 2 ** n_environment_qubits
    target_qubits_by_model: tuple[tuple[str, int], ...] = (
        ("iqc_zhangetal", 1),
        ("iqc_britoetal", 1),
        ("iqc_alfa", 1),
        ("iqc_multidimensional_2", 1),
        ("iqc_multidimensional_4", 1),
        ("iqc_multidimensional_8", 1),
        ("IQC", 1),
        ("IQC_AIL", 1),
        ("IQC_MULTI", 1),
        ("IQC_MULTI_2", 1),
        ("IQC_MULTI_4", 1),
        ("IQC_MULTI_8", 1),
    )
    alpha_vectors_by_model: tuple[tuple[str, tuple[tuple[float, float, float, float], ...]], ...] = ()

    def canonical_model(self, model: str) -> str:
        canonical = MODEL_ALIASES.get(model, model)
        if canonical not in MODEL_NAMES:
            raise ValueError(f"Unknown model {model!r}; choose from {MODEL_NAMES} plus aliases {tuple(MODEL_ALIASES)}.")
        return canonical

    def target_qubits(self, model: str) -> int:
        return dict(self.target_qubits_by_model)[model]

    def alpha_vectors(self, model: str) -> np.ndarray:
        vectors = dict(self.alpha_vectors_by_model).get(model)
        if vectors is None:
            return np.tile(np.array([[1.0, 1.0, 1.0, 1.0]], dtype=float), (self.target_qubits(model), 1))
        vectors = np.asarray(vectors, dtype=float)
        expected_shape = (self.target_qubits(model), 4)
        if vectors.shape != expected_shape:
            raise ValueError(f"alpha vectors for {model} must have shape {expected_shape}, got {vectors.shape}.")
        return vectors


def normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""

    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("A zero vector cannot define an input state.")
    return vector / norm


def _uniform_state(dimension: int) -> np.ndarray:
    return np.ones(dimension, dtype=complex) / np.sqrt(dimension)


def _kron_all(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("matrices cannot be empty")
    result = np.asarray(matrices[0], dtype=complex)
    for matrix in matrices[1:]:
        result = np.kron(result, np.asarray(matrix, dtype=complex))
    return result


def _alpha_matrix(alpha_vectors: np.ndarray, n_target_qubits: int, diagonal_p: bool) -> np.ndarray:
    alpha_matrix = np.asarray(alpha_vectors, dtype=float).reshape(n_target_qubits, 4).copy()
    if diagonal_p:
        # Preserve |(alpha_x, alpha_y, alpha_z)| and map direction to sigma_z.
        norms = np.linalg.norm(alpha_matrix[:, :3], axis=1)
        alpha_matrix[:, :3] = 0.0
        alpha_matrix[:, 2] = norms
    return alpha_matrix


def _sigma_q_from_alpha(alpha_matrix: np.ndarray) -> np.ndarray:
    locals_ = [np.asarray(get_weighted_sigmaQ_jnp(alpha), dtype=complex) for alpha in alpha_matrix]
    return _kron_all(locals_)


def _environment_dimension_for_model(model: str, n_features: int, fallback: int) -> int:
    if "multidimensional_2" in model:
        return 2
    if "multidimensional_4" in model or model == "IQC_MULTI":
        return 4
    if "multidimensional_8" in model:
        return 8
    if model in ("iqc_zhangetal", "iqc_britoetal", "iqc_alfa"):
        return n_features
    return int(fallback)


def _sample_weight_parameters(
    model: str,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample random model parameters with model-native shape."""

    n_e = _environment_dimension_for_model(model, config.n_features, config.environment_dimension)

    if model in ("iqc_zhangetal", "iqc_britoetal", "iqc_alfa"):
        return rng.random((config.n_pairs, config.n_features))

    if model in (
        "iqc_multidimensional_2",
        "iqc_multidimensional_4",
        "iqc_multidimensional_8",
    ):
        return rng.random((config.n_pairs, n_e, config.n_features))

    raise ValueError(f"Unknown model {model!r}; choose from {MODEL_NAMES}.")


def _sample_alpha_vectors(
    canonical_model: str,
    fixed_alpha_vectors: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    randomize_alpha: bool,
) -> np.ndarray:
    """Return alpha vectors per pair; keep zhang/brito fixed by design."""

    fixed_alpha_vectors = np.asarray(fixed_alpha_vectors, dtype=float)
    n_target_qubits = fixed_alpha_vectors.shape[0]

    if (not randomize_alpha) or (canonical_model in ALPHA_FIXED_MODELS):
        return np.repeat(fixed_alpha_vectors[np.newaxis, :, :], n_pairs, axis=0)

    return rng.random((n_pairs, n_target_qubits, 4))


def _unitary_and_environment_state(
    model: str,
    x: np.ndarray,
    weight_parameters: np.ndarray,
    alpha_matrix: np.ndarray,
    n_target_qubits: int,
    environment_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build unitary and environment ket according to each model family."""

    del n_target_qubits  # kept for readability in call sites

    input_x = np.asarray(x, dtype=float)
    sigma_q = _sigma_q_from_alpha(alpha_matrix)

    if model in ("iqc_zhangetal", "iqc_britoetal", "iqc_alfa"):
        weights = np.asarray(weight_parameters, dtype=float).reshape(-1)
        if weights.shape[0] != input_x.shape[0]:
            raise ValueError(
                f"{model} expects one weight per feature: got {weights.shape[0]} weights for {input_x.shape[0]} features."
            )

        if model == "iqc_britoetal":
            sigma_e = np.diag(weights)
            environment_state = normalize(input_x).astype(complex)
        else:
            sigma_e = np.diag(input_x * weights)
            environment_state = _uniform_state(len(input_x))

        unitary = np.asarray(get_U_operator_jax(sigma_q, sigma_e), dtype=complex)
        return unitary, environment_state

    if model in (
        "iqc_multidimensional_2",
        "iqc_multidimensional_4",
        "iqc_multidimensional_8",
    ):
        matrix_ws = np.asarray(weight_parameters, dtype=float)
        if matrix_ws.shape != (environment_dimension, input_x.shape[0]):
            raise ValueError(
                f"{model} expects weight matrix shape {(environment_dimension, input_x.shape[0])}, got {matrix_ws.shape}."
            )

        sigma_e = np.diag(matrix_ws @ input_x)
        environment_state = _uniform_state(environment_dimension)
        unitary = np.asarray(get_U_operator_jax(sigma_q, sigma_e), dtype=complex)
        return unitary, environment_state

    raise ValueError(f"Unknown model {model!r}; choose from {MODEL_NAMES}.")


def model_output_state(
    model: str,
    x: np.ndarray,
    weight_parameters: np.ndarray,
    n_target_qubits: int,
    alpha_vectors: np.ndarray | None = None,
    diagonal_p: bool = False,
) -> np.ndarray:
    """Return the global pure output state for one IQC parameter instance."""

    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; choose from {MODEL_NAMES}.")

    alpha_default = np.tile(np.array([[1.0, 1.0, 1.0, 1.0]], dtype=float), (n_target_qubits, 1))
    alpha_matrix = _alpha_matrix(
        alpha_vectors if alpha_vectors is not None else alpha_default,
        n_target_qubits=n_target_qubits,
        diagonal_p=diagonal_p,
    )

    environment_dimension = _environment_dimension_for_model(
        model,
        n_features=len(np.asarray(x, dtype=float)),
        fallback=int(np.asarray(weight_parameters).shape[0]) if np.asarray(weight_parameters).ndim > 1 else len(x),
    )

    unitary, environment_state = _unitary_and_environment_state(
        model=model,
        x=np.asarray(x, dtype=float),
        weight_parameters=np.asarray(weight_parameters, dtype=float),
        alpha_matrix=alpha_matrix,
        n_target_qubits=n_target_qubits,
        environment_dimension=environment_dimension,
    )

    cognitive_dimension = 2 ** n_target_qubits
    cognitive_state = _uniform_state(cognitive_dimension)
    return unitary @ np.kron(cognitive_state, environment_state)


def haar_bin_probabilities(dimension: int, bin_edges: np.ndarray) -> np.ndarray:
    """Exact Haar probabilities for pure-state fidelities in each bin."""

    lower, upper = bin_edges[:-1], bin_edges[1:]
    probabilities = (1 - lower) ** (dimension - 1) - (1 - upper) ** (dimension - 1)
    return probabilities / probabilities.sum()


def kl_from_fidelities(fidelities: np.ndarray, dimension: int, n_bins: int = 75) -> dict:
    """Compute one empirical histogram and D_KL(P || P_Haar)."""

    fidelities = np.asarray(fidelities, dtype=float)
    if np.any((fidelities < -1e-12) | (fidelities > 1 + 1e-12)):
        raise ValueError("Fidelities must be between zero and one.")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    empirical, _ = np.histogram(np.clip(fidelities, 0, 1), bins=bin_edges)
    empirical = empirical / empirical.sum()
    haar = haar_bin_probabilities(dimension, bin_edges)
    mask = empirical > 0
    kld = float(np.sum(empirical[mask] * np.log(empirical[mask] / haar[mask])))

    return {
        "kld": kld,
        "fidelities": fidelities,
        "empirical": empirical,
        "haar": haar,
        "bin_edges": bin_edges,
    }


def calculate_kld_byU(unitary_pairs, initial_state: np.ndarray, n_bins: int = 75) -> dict:
    """Validated unitary-only version of calculate_kld_byU."""

    initial_state = np.asarray(initial_state, dtype=complex)
    dimension = len(initial_state)
    fidelities = np.array([
        abs(np.vdot(u2 @ initial_state, u1 @ initial_state)) ** 2
        for u1, u2 in unitary_pairs
    ])
    return kl_from_fidelities(fidelities, dimension, n_bins)


def evaluate_model(model: str, config: BenchmarkConfig, diagonal_p: bool = False) -> dict:
    """Evaluate one model with the common state-fidelity protocol."""

    canonical_model = config.canonical_model(model)
    n_target = config.target_qubits(model)
    fixed_alpha_vectors = config.alpha_vectors(model)

    rng = np.random.default_rng(config.seed)
    x1 = rng.random((config.n_pairs, config.n_features))
    x2 = rng.random((config.n_pairs, config.n_features))
    w1 = _sample_weight_parameters(canonical_model, config, rng)
    w2 = _sample_weight_parameters(canonical_model, config, rng)
    alpha1 = _sample_alpha_vectors(
        canonical_model=canonical_model,
        fixed_alpha_vectors=fixed_alpha_vectors,
        n_pairs=config.n_pairs,
        rng=rng,
        randomize_alpha=config.randomize_alpha,
    )
    alpha2 = _sample_alpha_vectors(
        canonical_model=canonical_model,
        fixed_alpha_vectors=fixed_alpha_vectors,
        n_pairs=config.n_pairs,
        rng=rng,
        randomize_alpha=config.randomize_alpha,
    )

    fidelities = np.empty(config.n_pairs)
    state_dimension = None

    for index in range(config.n_pairs):
        psi_1 = model_output_state(
            model=canonical_model,
            x=x1[index],
            weight_parameters=w1[index],
            n_target_qubits=n_target,
            alpha_vectors=alpha1[index],
            diagonal_p=diagonal_p,
        )
        psi_2 = model_output_state(
            model=canonical_model,
            x=x2[index],
            weight_parameters=w2[index],
            n_target_qubits=n_target,
            alpha_vectors=alpha2[index],
            diagonal_p=diagonal_p,
        )
        fidelities[index] = abs(np.vdot(psi_2, psi_1)) ** 2

        if state_dimension is None:
            state_dimension = len(psi_1)

    result = kl_from_fidelities(fidelities, state_dimension, config.n_bins)

    # Relative expressibility
    if canonical_model == "iqc_britoetal":
        # For IQC-AIL, the input is encoded in the initial environment
        # state. Therefore, even the idle circuit generates different
        # states for different inputs.

        idle_fidelities = np.empty(config.n_pairs)

        for index in range(config.n_pairs):
            x1_norm = x1[index] / np.linalg.norm(x1[index])
            x2_norm = x2[index] / np.linalg.norm(x2[index])

            idle_fidelities[index] = abs(np.vdot(x2_norm, x1_norm)) ** 2

        idle_result = kl_from_fidelities(
            idle_fidelities,
            state_dimension,
            config.n_bins,
        )
        expr_idle = idle_result["kld"]

    else:
        # For the other models, the initial state is independent of x.
        # The idle circuit therefore produces the same state for every
        # realization, giving the analytical baseline.
        expr_idle = (state_dimension - 1) * np.log(config.n_bins)

    result["relative_expr"] = -np.log(result["kld"] / expr_idle)
    result["expr_idle"] = expr_idle

    result.update(
        {
            "model": model,
            "canonical_model": canonical_model,
            "n_target_qubits": n_target,
            "dimension": state_dimension,
            "diagonal_p": diagonal_p,
            "randomize_alpha": config.randomize_alpha and (canonical_model not in ALPHA_FIXED_MODELS),
        }
    )
    return result


def run_benchmark(config: BenchmarkConfig, models: tuple[str, ...] = MODEL_NAMES) -> list[dict]:
    """Run a fair comparison using identical sampling and histogram settings."""

    return [evaluate_model(model, config) for model in models]


def haar_sanity_check(dimension: int, n_pairs: int = 10_000, n_bins: int = 75, seed: int = 2026) -> dict:
    """Reference check: random Haar states should yield KL close to zero."""

    rng = np.random.default_rng(seed)
    left = rng.normal(size=(n_pairs, dimension)) + 1j * rng.normal(size=(n_pairs, dimension))
    right = rng.normal(size=(n_pairs, dimension)) + 1j * rng.normal(size=(n_pairs, dimension))
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    fidelities = np.abs(np.sum(np.conj(right) * left, axis=1)) ** 2
    return kl_from_fidelities(fidelities, dimension, n_bins)


__all__ = [
    "BASE_MODEL_NAMES",
    "MODEL_NAMES",
    "MODEL_ALIASES",
    "BenchmarkConfig",
    "evaluate_model",
    "run_benchmark",
    "haar_sanity_check",
    "kl_from_fidelities",
    "calculate_kld_byU",
]

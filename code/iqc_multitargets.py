"""Versões multi-target dos modelos IQC_DE e IQC multidimensional.

O sistema alvo possui ``N_qubits_tgt`` qubits. Para cada qubit são usados
três parâmetros em ``vector_alpha``. O vetor linear é reorganizado para
``(N_qubits_tgt, 3)``.

O operador do sistema alvo é construído como

    sigma_Q = sigma_Q^(0) ⊗ ... ⊗ sigma_Q^(N_qubits_tgt-1)

e a observável final é

    Z_target = Z ⊗ ... ⊗ Z.
"""

import jax.numpy as jnp

from iqc_zhangetal import (
    get_sigmaQ_from_polar_coord,
    get_U_operator_jax,
    get_weighted_sigmaQ_jnp,
    normalize,
)


def next_power_of_two(value):
    value = int(value)
    if value < 1:
        raise ValueError("A quantidade de features deve ser maior que zero.")
    return 1 << (value - 1).bit_length()


def _kron_all(operators):
    """Produto de Kronecker de uma sequência não vazia de matrizes."""
    if not operators:
        raise ValueError("A lista de operadores não pode ser vazia.")

    result = jnp.asarray(operators[0], dtype=jnp.complex64)
    for operator in operators[1:]:
        result = jnp.kron(
            result,
            jnp.asarray(operator, dtype=jnp.complex64),
        )
    return result


def _reshape_alpha(vector_alpha, N_qubits_tgt):
    """Valida e reorganiza o vetor alfa linear para (N_qubits_tgt, 4)."""
    N_qubits_tgt = int(N_qubits_tgt)
    if N_qubits_tgt < 1:
        raise ValueError("N_qubits_tgt deve ser maior ou igual a 1.")

    vector_alpha = jnp.asarray(
        vector_alpha,
        dtype=jnp.float32,
    ).reshape(-1)

    expected = 4 * N_qubits_tgt
    if int(vector_alpha.shape[0]) != expected:
        raise ValueError(
            "Quantidade inválida de parâmetros em vector_alpha.\n"
            f"N_qubits_tgt: {N_qubits_tgt}\n"
            f"Esperado: 4 * N_qubits_tgt = {expected}\n"
            f"Recebido: {int(vector_alpha.shape[0])}"
        )

    return vector_alpha.reshape(N_qubits_tgt, 4)


def _resolve_sigma_q_params(
    alpha_matrix,
    dic_classifier_params,
    N_qubits_tgt,
):
    """Obtém parâmetros de sigma_Q e garante shape (N_qubits_tgt, 4)."""
    supplied = dic_classifier_params.get("sigma_q_params", None)
    if supplied is None:
        return alpha_matrix

    supplied = jnp.asarray(supplied, dtype=jnp.float32)
    if int(supplied.size) != 4 * N_qubits_tgt:
        raise ValueError(
            "dic_classifier_params['sigma_q_params'] deve conter "
            f"{4 * N_qubits_tgt} valores para {N_qubits_tgt} qubits alvo."
        )
    return supplied.reshape(N_qubits_tgt, 4)


def _build_multi_target_sigma_q(
    alpha_matrix,
    use_polar_coordinates_on_sigma_q,
):
    """Constrói sigma_Q total por produto tensorial dos operadores locais."""
    local_operators = []

    for qubit_index in range(int(alpha_matrix.shape[0])):
        params = alpha_matrix[qubit_index]

        if use_polar_coordinates_on_sigma_q:
            sigma_local = get_sigmaQ_from_polar_coord(params)
        else:
            sigma_local = get_weighted_sigmaQ_jnp(params)

        local_operators.append(
            jnp.asarray(sigma_local, dtype=jnp.complex64)
        )

    return _kron_all(local_operators)


def _plus_density_matrix(N_qubits_tgt):
    """Retorna |+><+| tensorial para o sistema alvo completo."""
    target_dim = 2 ** int(N_qubits_tgt)
    ket = (
        jnp.ones((target_dim, 1), dtype=jnp.complex64)
        / jnp.sqrt(jnp.asarray(target_dim, dtype=jnp.float32))
    )
    return ket @ jnp.conj(ket).T


def _z_tensor(N_qubits_tgt):
    """Retorna Z^{tensor N_qubits_tgt}."""
    pauli_z = jnp.array(
        [[1.0, 0.0], [0.0, -1.0]],
        dtype=jnp.complex64,
    )
    return _kron_all([pauli_z] * int(N_qubits_tgt))


def get_iqc_de_multi_target_dimensions(n_features, N_qubits_tgt):
    """Dimensões e quantidades de parâmetros do IQC_DE multi-target."""
    N_e = next_power_of_two(n_features)
    N_qubits_tgt = int(N_qubits_tgt)
    if N_qubits_tgt < 1:
        raise ValueError("N_qubits_tgt deve ser maior ou igual a 1.")

    return {
        "original_features": int(n_features),
        "padded_features": N_e,
        "N_e": N_e,
        "N_qubits_tgt": N_qubits_tgt,
        "target_dimension": 2 ** N_qubits_tgt,
        "number_of_alpha_params": 4 * N_qubits_tgt,
        "number_of_weight_params": 1 + N_e * N_e,
    }


def get_iqc_multidimensional_multi_target_dimensions(
    n_features,
    N_e,
    N_qubits_tgt,
):
    """Dimensões e parâmetros do IQC multidimensional multi-target."""
    n_features = int(n_features)
    N_e = int(N_e)
    N_qubits_tgt = int(N_qubits_tgt)

    if min(n_features, N_e, N_qubits_tgt) < 1:
        raise ValueError("n_features, N_e e N_qubits_tgt devem ser positivos.")

    return {
        "original_features": n_features,
        "N_e": N_e,
        "N_qubits_tgt": N_qubits_tgt,
        "target_dimension": 2 ** N_qubits_tgt,
        "number_of_alpha_params": 4 * N_qubits_tgt,
        "number_of_weight_params": 1 + N_e * n_features,
    }


def iqc_de_multi_target(
    vector_x,
    vector_alpha,
    vector_ws,
    N_e=None,
    normalize_x=False,
    normalize_w=False,
    dic_classifier_params=None,
    N_qubits=None,
    N_qubits_tgt=1,
    load_inputvector_env_state=True,
    padding_value=0.1,
):
    """IQC Double Encoding com múltiplos qubits no sistema alvo.

    ``vector_alpha`` é linear e deve possuir exatamente
    ``4 * N_qubits_tgt`` entradas. Internamente:

        alpha_matrix = vector_alpha.reshape(N_qubits_tgt, 4)

    O sistema alvo tem dimensão ``2**N_qubits_tgt``. O traço parcial elimina
    apenas o ambiente e a medição é feita com ``Z**tensor(N_qubits_tgt)``.

    Os pesos continuam parametrizando somente sigma_E:

        vector_ws = [bias, W.flatten()]
        W.shape = (N_e, N_e)
    """
    del N_qubits  # Mantido apenas por compatibilidade de assinatura.

    if dic_classifier_params is None:
        dic_classifier_params = {}

    N_qubits_tgt = int(N_qubits_tgt)
    alpha_matrix = _reshape_alpha(vector_alpha, N_qubits_tgt)

    vector_x = jnp.asarray(vector_x, dtype=jnp.float32).reshape(-1)
    vector_ws = jnp.asarray(vector_ws, dtype=jnp.float32).reshape(-1)

    n_features_original = int(vector_x.shape[0])
    if n_features_original < 1:
        raise ValueError("vector_x deve possuir pelo menos uma feature.")

    calculated_N_e = next_power_of_two(n_features_original)
    if N_e is not None and int(N_e) != calculated_N_e:
        raise ValueError(
            "O N_e informado é incompatível com o cálculo automático.\n"
            f"N_e informado: {N_e}; N_e esperado: {calculated_N_e}."
        )
    N_e = calculated_N_e

    padding_size = N_e - n_features_original
    if padding_size > 0:
        vector_x = jnp.pad(
            vector_x,
            pad_width=(0, padding_size),
            mode="constant",
            constant_values=padding_value,
        )

    expected_number_of_params = 1 + N_e * N_e
    if int(vector_ws.shape[0]) != expected_number_of_params:
        raise ValueError(
            "Quantidade inválida de parâmetros para IQC_DE multi-target.\n"
            f"Esperado: {expected_number_of_params}\n"
            f"Recebido: {int(vector_ws.shape[0])}"
        )

    bias = vector_ws[0]
    flat_weights = vector_ws[1:]

    if normalize_x:
        vector_x = normalize(vector_x)

    if dic_classifier_params.get("use_exponential_on_input", False):
        vector_x = jnp.exp(vector_x)

    sigma_q_params = _resolve_sigma_q_params(
        alpha_matrix,
        dic_classifier_params,
        N_qubits_tgt,
    )
    sigmaQ = _build_multi_target_sigma_q(
        sigma_q_params,
        dic_classifier_params.get(
            "use_polar_coordinates_on_sigma_q",
            False,
        ),
    )

    matrix_ws = flat_weights.reshape(N_e, N_e)
    if normalize_w:
        norms = jnp.linalg.norm(matrix_ws, axis=1, keepdims=True)
        matrix_ws = matrix_ws / jnp.maximum(norms, 1e-12)

    sigmaE_diagonal = matrix_ws @ vector_x
    sigmaE = jnp.diag(
        sigmaE_diagonal.astype(jnp.complex64)
    )

    if load_inputvector_env_state:
        amplitude_norm = jnp.linalg.norm(vector_x)
        uniform_state = (
            jnp.ones((N_e,), dtype=jnp.float32)
            / jnp.sqrt(jnp.asarray(N_e, dtype=jnp.float32))
        )
        normalized_amplitudes = jnp.where(
            amplitude_norm > 1e-12,
            vector_x / amplitude_norm,
            uniform_state,
        )
        ket_env = normalized_amplitudes.reshape(N_e, 1).astype(jnp.complex64)
    else:
        ket_env = (
            jnp.ones((N_e, 1), dtype=jnp.complex64)
            / jnp.sqrt(jnp.asarray(N_e, dtype=jnp.float32))
        )

    p_env = ket_env @ jnp.conj(ket_env).T
    p_target = _plus_density_matrix(N_qubits_tgt)
    target_dim = 2 ** N_qubits_tgt

    U_operator = get_U_operator_jax(sigmaQ, sigmaE)
    p_target_env = jnp.kron(p_target, p_env)
    p_out = (
        U_operator
        @ p_target_env
        @ jnp.conj(U_operator).T
    )

    p_target_new = jnp.trace(
        p_out.reshape(target_dim, N_e, target_dim, N_e),
        axis1=1,
        axis2=3,
    )

    observable = _z_tensor(N_qubits_tgt)
    expectation = jnp.real(
        jnp.trace(p_target_new @ observable)
    )

    return expectation + bias


def iqc_multidimensional_multi_target(
    vector_x,
    vector_alpha,
    vector_ws,
    N_e,
    normalize_x=False,
    normalize_w=False,
    dic_classifier_params=None,
    N_qubits=None,
    N_qubits_tgt=1,
    load_inputvector_env_state=False,
):
    """IQC multidimensional com múltiplos qubits no sistema alvo.

    ``vector_alpha`` deve conter ``4 * N_qubits_tgt`` valores e é
    reorganizado para ``(N_qubits_tgt, 4)``.

    Os pesos preservam a parametrização original:

        vector_ws = [bias, W.flatten()]
        W.shape = (N_e, n_features)
    """
    del N_qubits  # Mantido apenas por compatibilidade de assinatura.

    if dic_classifier_params is None:
        dic_classifier_params = {}

    N_e = int(N_e)
    if N_e < 1:
        raise ValueError("N_e deve ser maior ou igual a 1.")

    N_qubits_tgt = int(N_qubits_tgt)
    alpha_matrix = _reshape_alpha(vector_alpha, N_qubits_tgt)

    vector_x = jnp.asarray(vector_x, dtype=jnp.float32).reshape(-1)
    vector_ws = jnp.asarray(vector_ws, dtype=jnp.float32).reshape(-1)
    N_features = int(vector_x.shape[0])

    if N_features < 1:
        raise ValueError("vector_x deve possuir pelo menos uma feature.")

    expected_number_of_params = 1 + N_e * N_features
    if int(vector_ws.shape[0]) != expected_number_of_params:
        raise ValueError(
            "Quantidade inválida de parâmetros para o IQC multidimensional "
            "multi-target.\n"
            f"Esperado: 1 + N_e * n_features = {expected_number_of_params}\n"
            f"Recebido: {int(vector_ws.shape[0])}"
        )

    bias = vector_ws[0]
    flat_weights = vector_ws[1:]

    if normalize_x:
        vector_x = normalize(vector_x)

    if dic_classifier_params.get("use_exponential_on_input", False):
        vector_x = jnp.exp(vector_x)

    sigma_q_params = _resolve_sigma_q_params(
        alpha_matrix,
        dic_classifier_params,
        N_qubits_tgt,
    )
    sigmaQ = _build_multi_target_sigma_q(
        sigma_q_params,
        dic_classifier_params.get(
            "use_polar_coordinates_on_sigma_q",
            False,
        ),
    )

    matrix_ws = flat_weights.reshape(N_e, N_features)
    if normalize_w:
        norms = jnp.linalg.norm(matrix_ws, axis=1, keepdims=True)
        matrix_ws = matrix_ws / jnp.maximum(norms, 1e-12)

    sigmaE_diagonal = matrix_ws @ vector_x
    sigmaE = jnp.diag(
        sigmaE_diagonal.astype(jnp.complex64)
    )

    if load_inputvector_env_state:
        raise ValueError(
            "iqc_multidimensional_multi_target não usa "
            "load_inputvector_env_state=True."
        )

    ket_env = (
        jnp.ones((N_e, 1), dtype=jnp.complex64)
        / jnp.sqrt(jnp.asarray(N_e, dtype=jnp.float32))
    )
    p_env = ket_env @ jnp.conj(ket_env).T

    p_target = _plus_density_matrix(N_qubits_tgt)
    target_dim = 2 ** N_qubits_tgt

    U_operator = get_U_operator_jax(sigmaQ, sigmaE)
    p_target_env = jnp.kron(p_target, p_env)
    p_out = (
        U_operator
        @ p_target_env
        @ jnp.conj(U_operator).T
    )

    p_target_new = jnp.trace(
        p_out.reshape(target_dim, N_e, target_dim, N_e),
        axis1=1,
        axis2=3,
    )

    observable = _z_tensor(N_qubits_tgt)
    expectation = jnp.real(
        jnp.trace(p_target_new @ observable)
    )

    return expectation + bias


# Alias para tolerar a grafia usada no pedido sem propagar o erro no nome oficial.
iqc_multimendional_multi_target = iqc_multidimensional_multi_target
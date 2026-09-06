import jax.numpy as jnp

from iqc_zhangetal import (
    get_sigmaQ_from_polar_coord,
    get_U_operator_jax,
    get_weighted_sigmaQ_jnp,
    normalize,
)


def iqc_multidimensional(
    vector_x,
    vector_alpha,
    vector_ws,
    N_e,
    normalize_x=False,
    normalize_w=False,
    dic_classifier_params=None,
    N_qubits=None,
    N_qubits_tgt=None,
    load_inputvector_env_state=False,
):
    """
    IQC multidimensional baseado em iqc_zhangetal.

    Parametrização:
        vector_ws = [bias, w_1, ..., w_Ne]

    onde cada w_i possui a mesma dimensão de vector_x. Assim,

        diag(sigma_E)_i = <w_i, x>.

    A quantidade total de parâmetros treináveis é:

        1 + N_e * len(vector_x).
    """
    c1 = vector_alpha[0]
    c2 = vector_alpha[1]
    c3 = vector_alpha[2]
    c4 = vector_alpha[3]
    bias = vector_ws[0]
    vector_ws = vector_ws[1:]

    if dic_classifier_params is None:
        dic_classifier_params = {}

    vector_x = jnp.asarray(vector_x, dtype=jnp.float32)
    N_features = vector_x.shape[0]

    expected_weights = N_e * N_features
    if vector_ws.shape[0] != expected_weights:
        raise ValueError(
            "Quantidade inválida de pesos: esperado "
            f"1 + N_e * n_features = 1 + {N_e} * {N_features} = "
            f"{1 + expected_weights} parâmetros, mas foram recebidos "
            f"{1 + vector_ws.shape[0]}."
        )

    sigma_q_params = dic_classifier_params.get(
        "sigma_q_params",
        [c1, c2, c3, c4],
    )
    use_polar_coordinates_on_sigma_q = dic_classifier_params.get(
        "use_polar_coordinates_on_sigma_q",
        False,
    )

    if normalize_x:
        vector_x = normalize(vector_x)
    if dic_classifier_params.get("use_exponential_on_input", False):
        vector_x = jnp.exp(vector_x)

    if use_polar_coordinates_on_sigma_q:
        sigmaQ = get_sigmaQ_from_polar_coord(sigma_q_params)
    else:
        sigmaQ = get_weighted_sigmaQ_jnp(sigma_q_params)

    sigmaQ = jnp.asarray(sigmaQ, dtype=jnp.complex64)

    # Cada linha representa um vetor w_i com dim(w_i) = dim(x).
    matrix_ws = jnp.asarray(vector_ws, dtype=jnp.float32).reshape(
        N_e,
        N_features,
    )

    if normalize_w:
        matrix_ws = matrix_ws / (
            jnp.linalg.norm(matrix_ws, axis=1, keepdims=True) + 1e-16
        )

    # Nova definição: uma entrada diagonal para cada produto interno w_i^T x.
    sigmaE_diagonal = matrix_ws @ vector_x
    sigmaE = jnp.diag(sigmaE_diagonal)

    # O ambiente passa a ter dimensão N_e, a mesma dimensão de sigmaE.
    p_env = jnp.ones((N_e, 1), dtype=jnp.float32) / jnp.sqrt(N_e)
    p_env = p_env @ p_env.T

    p_cog = jnp.ones((2, 1), dtype=jnp.float32) / jnp.sqrt(2)
    p_cog = p_cog @ p_cog.T

    # Mantido apenas por compatibilidade de assinatura. O novo modelo codifica
    # x em cada produto interno w_i^T x que compõe sigmaE.
    if load_inputvector_env_state:
        raise ValueError(
            "iqc_multidimensional não usa load_inputvector_env_state=True, "
            "pois a dimensão do ambiente é N_e e cada entrada de sigmaE é w_i^T x."
        )

    U_operator = get_U_operator_jax(sigmaQ, sigmaE)

    p_cog_env = jnp.kron(p_cog, p_env)
    p_out = U_operator @ p_cog_env @ jnp.conj(U_operator).T
    p_cog_new = jnp.trace(
        p_out.reshape([2, N_e, 2, N_e]),
        axis1=1,
        axis2=3,
    )

    pauli_z = jnp.array(
        [[1, 0], [0, -1]],
        dtype=jnp.complex64,
    )
    expectation = jnp.real(jnp.trace(p_cog_new @ pauli_z))

    return expectation + bias
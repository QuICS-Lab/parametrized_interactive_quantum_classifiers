import jax.numpy as jnp

from iqc_zhangetal import (
    get_sigmaQ_from_polar_coord,
    get_U_operator_jax,
    get_weighted_sigmaQ_jnp,
    normalize,
)


def next_power_of_two(value):
    """
    Retorna a menor potência de 2 maior ou igual a value.

    Exemplos:
        1 -> 1
        2 -> 2
        3 -> 4
        4 -> 4
        5 -> 8
        13 -> 16
    """
    value = int(value)

    if value < 1:
        raise ValueError(
            "A quantidade de features deve ser maior que zero."
        )

    return 1 << (value - 1).bit_length()


def get_iqc_de_dimensions(n_features):
    """
    Calcula automaticamente:

        N_e
        n_features_padded
        number_of_params

    No IQC_DE, após o padding:

        N_e = n_features_padded

    e a quantidade de parâmetros treináveis é:

        1 + N_e²
    """
    N_e = next_power_of_two(n_features)

    return {
        "original_features": int(n_features),
        "padded_features": N_e,
        "N_e": N_e,
        "number_of_params": 1 + N_e * N_e,
    }


def iqc_de(
    vector_x,
    vector_alpha,
    vector_ws,
    N_e=None,
    normalize_x=False,
    normalize_w=False,
    dic_classifier_params=None,
    N_qubits=None,
    N_qubits_tgt=None,
    load_inputvector_env_state=True,
    padding_value=0.1,
):
    """
    IQC Double Encoding — IQC_DE.

    O vetor x é codificado simultaneamente:

    1. No estado inicial do ambiente, por amplitude loading.
    2. No operador sigma_E, pelos produtos internos w_i^T x.

    A dimensão do ambiente é calculada automaticamente como a
    menor potência de 2 maior ou igual à quantidade original
    de features:

        N_e = 2^ceil(log2(n_features))

    Caso n_features não seja potência de 2, vector_x é preenchido
    com padding_value até possuir dimensão N_e.

    Exemplo
    -------
    Para:

        vector_x = [x1, x2, x3]
        padding_value = 0.1

    temos:

        N_e = 4
        vector_x_padded = [x1, x2, x3, 0.1]

    Parametrização
    --------------
    Após o padding:

        dim(x_padded) = N_e

    Cada vetor w_i também possui dimensão N_e:

        W.shape = (N_e, N_e)

    Assim:

        sigma_E(x, W) = diag(W @ x_padded)

    A quantidade total de parâmetros treináveis é:

        1 + N_e²

    correspondendo a:

        [bias, W.flatten()]
    """

    if dic_classifier_params is None:
        dic_classifier_params = {}

    # ========================================================
    # Entrada original
    # ========================================================

    vector_x = jnp.asarray(
        vector_x,
        dtype=jnp.float32,
    ).reshape(-1)

    vector_alpha = jnp.asarray(
        vector_alpha,
        dtype=jnp.float32,
    ).reshape(-1)

    vector_ws = jnp.asarray(
        vector_ws,
        dtype=jnp.float32,
    ).reshape(-1)

    n_features_original = int(vector_x.shape[0])

    if n_features_original < 1:
        raise ValueError(
            "vector_x deve possuir pelo menos uma feature."
        )

    # ========================================================
    # Cálculo automático de N_e
    # ========================================================

    calculated_N_e = next_power_of_two(
        n_features_original
    )

    # N_e permanece na assinatura apenas para compatibilidade
    # com a classe IQC e versões anteriores.
    if N_e is not None and int(N_e) != calculated_N_e:
        raise ValueError(
            "O N_e informado é incompatível com o cálculo "
            "automático do IQC_DE.\n"
            f"n_features original: {n_features_original}\n"
            f"N_e informado: {N_e}\n"
            f"N_e esperado: {calculated_N_e}"
        )

    N_e = calculated_N_e

    # ========================================================
    # Padding de x
    # ========================================================

    padding_size = N_e - n_features_original

    if padding_size > 0:
        vector_x = jnp.pad(
            vector_x,
            pad_width=(0, padding_size),
            mode="constant",
            constant_values=padding_value,
        )

    # Agora:
    #
    # vector_x.shape == (N_e,)
    n_features_padded = int(vector_x.shape[0])

    if n_features_padded != N_e:
        raise RuntimeError(
            "Erro interno no padding: "
            f"dim(x_padded)={n_features_padded}, "
            f"mas N_e={N_e}."
        )

    # ========================================================
    # Validação dos parâmetros
    # ========================================================

    if vector_alpha.shape[0] < 4:
        raise ValueError(
            "vector_alpha deve possuir pelo menos "
            "quatro parâmetros."
        )

    expected_number_of_params = 1 + N_e * N_e

    if vector_ws.shape[0] != expected_number_of_params:
        raise ValueError(
            "Quantidade inválida de parâmetros para IQC_DE.\n"
            f"Features originais: {n_features_original}\n"
            f"Features após padding: {n_features_padded}\n"
            f"N_e: {N_e}\n"
            f"Esperado: 1 + N_e² = "
            f"1 + {N_e}² = {expected_number_of_params}\n"
            f"Recebido: {vector_ws.shape[0]}"
        )

    # ========================================================
    # Parâmetros de sigma_Q
    # ========================================================

    c1 = vector_alpha[0]
    c2 = vector_alpha[1]
    c3 = vector_alpha[2]
    c4 = vector_alpha[3]
    bias = vector_ws[0]
    flat_weights = vector_ws[1:]

    sigma_q_params = dic_classifier_params.get(
        "sigma_q_params",
        [c1, c2, c3, c4],
    )

    use_polar_coordinates_on_sigma_q = (
        dic_classifier_params.get(
            "use_polar_coordinates_on_sigma_q",
            False,
        )
    )

    # ========================================================
    # Pré-processamento de x
    # ========================================================

    if normalize_x:
        vector_x = normalize(vector_x)

    if dic_classifier_params.get(
        "use_exponential_on_input",
        False,
    ):
        vector_x = jnp.exp(vector_x)

    # ========================================================
    # Construção de sigma_Q
    # ========================================================

    if use_polar_coordinates_on_sigma_q:
        sigmaQ = get_sigmaQ_from_polar_coord(
            sigma_q_params
        )
    else:
        sigmaQ = get_weighted_sigmaQ_jnp(
            sigma_q_params
        )

    sigmaQ = jnp.asarray(
        sigmaQ,
        dtype=jnp.complex64,
    )

    # ========================================================
    # Primeira codificação:
    # x no operador sigma_E
    # ========================================================

    # Cada linha é um vetor w_i.
    #
    # matrix_ws.shape = (N_e, N_e)
    matrix_ws = flat_weights.reshape(
        N_e,
        N_e,
    )

    if normalize_w:
        weight_norms = jnp.linalg.norm(
            matrix_ws,
            axis=1,
            keepdims=True,
        )

        matrix_ws = matrix_ws / jnp.maximum(
            weight_norms,
            1e-12,
        )

    # Cada posição diagonal recebe:
    #
    # sigmaE_i = w_i^T x_padded
    sigmaE_diagonal = matrix_ws @ vector_x

    sigmaE = jnp.diag(
        sigmaE_diagonal.astype(jnp.complex64)
    )

    # ========================================================
    # Segunda codificação:
    # amplitude loading no estado inicial do ambiente
    # ========================================================

    if load_inputvector_env_state:
        amplitude_norm = jnp.linalg.norm(
            vector_x
        )

        uniform_state = (
            jnp.ones(
                (N_e,),
                dtype=jnp.float32,
            )
            / jnp.sqrt(
                jnp.asarray(
                    N_e,
                    dtype=jnp.float32,
                )
            )
        )

        # Caso x seja o vetor nulo, utiliza estado uniforme.
        normalized_amplitudes = jnp.where(
            amplitude_norm > 1e-12,
            vector_x / amplitude_norm,
            uniform_state,
        )

        ket_env = normalized_amplitudes.reshape(
            N_e,
            1,
        ).astype(jnp.complex64)

    else:
        ket_env = (
            jnp.ones(
                (N_e, 1),
                dtype=jnp.complex64,
            )
            / jnp.sqrt(
                jnp.asarray(
                    N_e,
                    dtype=jnp.float32,
                )
            )
        )

    p_env = ket_env @ jnp.conj(ket_env).T

    # ========================================================
    # Estado inicial do sistema cognitivo
    # ========================================================

    ket_cog = (
        jnp.ones(
            (2, 1),
            dtype=jnp.complex64,
        )
        / jnp.sqrt(
            jnp.asarray(
                2,
                dtype=jnp.float32,
            )
        )
    )

    p_cog = ket_cog @ jnp.conj(ket_cog).T

    # ========================================================
    # Evolução conjunta
    # ========================================================

    U_operator = get_U_operator_jax(
        sigmaQ,
        sigmaE,
    )

    p_cog_env = jnp.kron(
        p_cog,
        p_env,
    )

    p_out = (
        U_operator
        @ p_cog_env
        @ jnp.conj(U_operator).T
    )

    # ========================================================
    # Traço parcial sobre o ambiente
    # ========================================================

    p_cog_new = jnp.trace(
        p_out.reshape(
            2,
            N_e,
            2,
            N_e,
        ),
        axis1=1,
        axis2=3,
    )

    # ========================================================
    # Valor esperado de Pauli-Z
    # ========================================================

    pauli_z = jnp.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=jnp.complex64,
    )

    expectation = jnp.real(
        jnp.trace(
            p_cog_new @ pauli_z
        )
    )

    return expectation + bias




def next_power_of_two(n):
    return 1 << (int(n) - 1).bit_length()


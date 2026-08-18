

import jax
from jax import numpy as jnp
import optax
import jax
from jax import numpy as jnp
import optax
import numpy as np


def iqc_zhangetal(
    vector_x,
    vector_alpha,
    vector_ws,
    normalize_x=False,
    normalize_w=False,
    dic_classifier_params=None,
    N_qubits=None,
    N_qubits_tgt=None,
    load_inputvector_env_state=False #Brito et al. Model (amplitude encoding of information)
):
    """
    Core IQC-based regressor (inference path):
    - Builds sigma_Q, sigma_E
    - Evolves ρ_cog ⊗ ρ_env via U
    - Takes partial trace over environment
    - Returns expectation value of Pauli-Z as regressed output.
    """

    
    c1 = vector_alpha[0]
    c2 = vector_alpha[1]
    c3 = vector_alpha[2]
    c4 = vector_alpha[3]
    bias = vector_ws[0]
    vector_ws = vector_ws[1:]  # Remove bias from weights

    if dic_classifier_params is None:
        dic_classifier_params = {}

    N = len(vector_x)

    sigma_q_params = dic_classifier_params.get("sigma_q_params", [c1,c2,c3,c4])
    use_polar_coordinates_on_sigma_q = dic_classifier_params.get(
        "use_polar_coordinates_on_sigma_q", False
    )

    if normalize_x:
        vector_x = normalize(vector_x)
    if dic_classifier_params.get("use_exponential_on_input", False):
        vector_x = np.exp(vector_x)

    if use_polar_coordinates_on_sigma_q:
        sigmaQ = get_sigmaQ_from_polar_coord(sigma_q_params)
    else:
        sigmaQ = get_weighted_sigmaQ_jnp(sigma_q_params)

    sigmaQ = jnp.array(sigmaQ, dtype=jnp.complex64)

    vector_x = jnp.array(vector_x, dtype=jnp.float32)

    p_env = jnp.ones((N, 1)) / jnp.sqrt(N)
    p_env = p_env @ p_env.T

    p_cog = jnp.ones((2, 1)) / jnp.sqrt(2)
    p_cog = p_cog @ p_cog.T

    ##vector_ws_jnp = jnp.array(vector_ws, dtype=jnp.float32) #[jnp.array(w, dtype=jnp.float32) for w in vector_ws]
    vector_ws = jnp.array(vector_ws, dtype=jnp.float32) #[jnp.array(w, dtype=jnp.float32) for w in vector_ws]

    p_cog_new = p_cog
    U_operators = []
    p_out = None


    if normalize_w:
        vector_w = vector_w / (jnp.linalg.norm(vector_w) + 1e-16)

    # Equivalent to Eq #15
    if load_inputvector_env_state:

        # We can either keep only weights (in case we have only one environment)
        #sigmaE = jnp.diag(jnp.array(vector_w, dtype=jnp.float32))
        sigmaE = jnp.diag(vector_ws)
    else:
        # Or keep both as the original ICQ article
        #sigmaE = get_sigmaE(vector_x, vector_w, dic_classifier_params)
        sigmaE = jnp.diag(jnp.multiply(vector_x , vector_ws))

    # Eq #19 applied on a Quantum state equivalent of Hadamard(|00...0>) = 1/sqrt(N) * (|00...0> + ... + |11...1>)
    if load_inputvector_env_state:
        # We can either have Hadamard applied to each instance attribute...
        vector_x_norm = (jnp.linalg.norm(vector_x) + 1e-16)

        # env = x1/norm(x) |0> + x2/norm(x) |1> .... + xn/norm(x) |n>
        p_env = jnp.array(vector_x).reshape((N, 1)) / vector_x_norm
        p_env = p_env @ p_env.T


    #sigmaE = jnp.diag(vector_x * vector_w)
    U_operator = get_U_operator_jax(sigmaQ, sigmaE)
    U_operators.append(U_operator)

    p_cog_env = jnp.kron(p_cog_new, p_env)
    p_out = U_operator @ p_cog_env @ jnp.conj(U_operator).T
    p_cog_new = jnp.trace(p_out.reshape([2, N, 2, N]), axis1=1, axis2=3)

    p_cog_new_00 = jnp.real(p_cog_new[0, 0])
    p_cog_new_11 = jnp.real(p_cog_new[1, 1])

    pauli_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    expectation = jnp.real(jnp.trace(p_cog_new @ pauli_z))

    output_dict = {
        "U_operators": U_operators,
        "p_00": p_cog_new_00,
        "p_11": p_cog_new_11,
    }

    #return expectation, p_cog_new_11, output_dict
    return expectation + bias



def iqc_britoetal(
    vector_x,
    vector_alpha,
    vector_ws,
    normalize_x=False,
    normalize_w=False,
    dic_classifier_params=None,
    N_qubits=None,
    N_qubits_tgt=None,
    load_inputvector_env_state=True #Brito et al. Model (amplitude encoding of information)
):
    """
    Core IQC-based regressor (inference path):
    - Builds sigma_Q, sigma_E
    - Evolves ρ_cog ⊗ ρ_env via U
    - Takes partial trace over environment
    - Returns expectation value of Pauli-Z as regressed output.
    """

    
    c1 = vector_alpha[0]
    c2 = vector_alpha[1]
    c3 = vector_alpha[2]
    c4 = vector_alpha[3]
    bias = vector_ws[0]
    vector_ws = vector_ws[1:]  # Remove bias from weights

    if dic_classifier_params is None:
        dic_classifier_params = {}

    N = len(vector_x)

    sigma_q_params = dic_classifier_params.get("sigma_q_params", [c1,c2,c3,c4])
    use_polar_coordinates_on_sigma_q = dic_classifier_params.get(
        "use_polar_coordinates_on_sigma_q", False
    )

    if normalize_x:
        vector_x = normalize(vector_x)
    if dic_classifier_params.get("use_exponential_on_input", False):
        vector_x = np.exp(vector_x)

    if use_polar_coordinates_on_sigma_q:
        sigmaQ = get_sigmaQ_from_polar_coord(sigma_q_params)
    else:
        sigmaQ = get_weighted_sigmaQ_jnp(sigma_q_params)

    sigmaQ = jnp.array(sigmaQ, dtype=jnp.complex64)

    vector_x = jnp.array(vector_x, dtype=jnp.float32)

    p_env = jnp.ones((N, 1)) / jnp.sqrt(N)
    p_env = p_env @ p_env.T

    p_cog = jnp.ones((2, 1)) / jnp.sqrt(2)
    p_cog = p_cog @ p_cog.T

    ##vector_ws_jnp = jnp.array(vector_ws, dtype=jnp.float32) #[jnp.array(w, dtype=jnp.float32) for w in vector_ws]
    vector_ws = jnp.array(vector_ws, dtype=jnp.float32) #[jnp.array(w, dtype=jnp.float32) for w in vector_ws]

    p_cog_new = p_cog
    U_operators = []
    p_out = None


    if normalize_w:
        vector_w = vector_w / (jnp.linalg.norm(vector_w) + 1e-16)

    # Equivalent to Eq #15
    if load_inputvector_env_state:

        # We can either keep only weights (in case we have only one environment)
        #sigmaE = jnp.diag(jnp.array(vector_w, dtype=jnp.float32))
        sigmaE = jnp.diag(vector_ws)
    else:
        # Or keep both as the original ICQ article
        #sigmaE = get_sigmaE(vector_x, vector_w, dic_classifier_params)
        sigmaE = jnp.diag(jnp.multiply(vector_x , vector_ws))

    # Eq #19 applied on a Quantum state equivalent of Hadamard(|00...0>) = 1/sqrt(N) * (|00...0> + ... + |11...1>)
    if load_inputvector_env_state:
        # We can either have Hadamard applied to each instance attribute...
        vector_x_norm = (jnp.linalg.norm(vector_x) + 1e-16)

        # env = x1/norm(x) |0> + x2/norm(x) |1> .... + xn/norm(x) |n>
        p_env = jnp.array(vector_x).reshape((N, 1)) / vector_x_norm
        p_env = p_env @ p_env.T


    #sigmaE = jnp.diag(vector_x * vector_w)
    U_operator = get_U_operator_jax(sigmaQ, sigmaE)
    U_operators.append(U_operator)

    p_cog_env = jnp.kron(p_cog_new, p_env)
    p_out = U_operator @ p_cog_env @ jnp.conj(U_operator).T
    p_cog_new = jnp.trace(p_out.reshape([2, N, 2, N]), axis1=1, axis2=3)

    p_cog_new_00 = jnp.real(p_cog_new[0, 0])
    p_cog_new_11 = jnp.real(p_cog_new[1, 1])

    pauli_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    expectation = jnp.real(jnp.trace(p_cog_new @ pauli_z))

    output_dict = {
        "U_operators": U_operators,
        "p_00": p_cog_new_00,
        "p_11": p_cog_new_11,
    }

    #return expectation, p_cog_new_11, output_dict
    return expectation + bias




def normalize(x):
    return x / (np.linalg.norm(x) + 1e-16)


def get_p(psi):
    """
    |psi><psi| density matrix.
    """
    psi = np.matrix(psi)
    return psi * psi.getH()


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
        + param[3] * identity ###WITHOUT Identity
    )
    sigmaq_trace = jnp.trace(sigmaQ) ###WITHOUT Identity
    #if sigmaq_trace > 0:
    #    return jnp.array(sigmaQ) / sigmaq_trace
    return jnp.array(sigmaQ) / (sigmaq_trace+0.000000000000000000001) 
    #return jnp.array(sigmaQ)

def get_weighted_sigmaQ(param, iqcpq=False):
    """
    Build sigma_Q.

    - If iqcpq=False: linear combination of Pauli matrices + identity (Eq. 16).
    - If iqcpq=True: builds an n-level Hermitian matrix with fixed diagonal/off-diagonal.
    """
    if iqcpq:
        n = len(param)
        diagonal = np.full(n, 1, dtype=complex)
        diagonal[-1] = -np.sum(diagonal[:-1])

        off_diagonal = np.full((n, n), 1 + 1j, dtype=complex)
        matrix = np.zeros((n, n), dtype=complex)
        np.fill_diagonal(matrix, diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = off_diagonal[i, j]
                matrix[j, i] = np.conj(off_diagonal[i, j])
        return matrix

    sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
    sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)
    ###identity = np.array([[1, 0], [0, 1]], dtype=complex) ###WITHOUT Identity

    sigmaQ = (
        param[0] * sigmaX
        + param[1] * sigmaY
        + param[2] * sigmaZ
        ###+ param[3] * identity ###WITHOUT Identity
    )
    ###sigmaq_trace = np.trace(sigmaQ)###WITHOUT Identity
    ###if sigmaq_trace > 0:
    ###    return np.array(sigmaQ) / sigmaq_trace ###WITHOUT Identity
    return np.array(sigmaQ)


def get_sigmaQ_from_polar_coord(param):
    """
    Polar-coordinates version of sigma_Q, normalized to trace 1.
    """
    r, theta, phi = param

    rx = r * np.sin(theta) * np.cos(phi)
    ry = r * np.sin(theta) * np.sin(phi)
    rz = r * np.cos(theta)

    sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
    sigmaY = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)

    return (identity + rx * sigmaX + ry * sigmaY + rz * sigmaZ) / 2.0



def get_sigmaE(vector_x, vector_w, dic_classifier_params, ndse=False):
    """
    Environment operator sigma_E (Eq. 17), JAX-compatible.
    """
    load_inputvector_env_state = dic_classifier_params.get(
        "load_inputvector_env_state", False
    )

    vx = jnp.atleast_1d(vector_x).flatten()
    vw = jnp.atleast_1d(vector_w).flatten()

    if load_inputvector_env_state:
        return jnp.diag(vx)
    # default: element-wise product on diagonal
    return jnp.diag(vx * vw)




import scipy
def get_U_operator_jax(sigmaQ, sigmaE):
    """
    U = exp(+i * (sigma_Q ⊗ sigma_E))  (Eq. 15/22 from the paper).
    Derived from H_int = -ℏg σ_Q⊗σ_E and U(t) = exp(-iH_int t/ℏ).
    """
    H = jnp.kron(sigmaQ, sigmaE)
    U = jax.scipy.linalg.expm(1j * H)
    return U

def get_U_operator(sigmaQ, sigmaE):
    """
    U = exp(+i * (sigma_Q ⊗ sigma_E))  (Eq. 15/22 from the paper).
    Derived from H_int = -ℏg σ_Q⊗σ_E and U(t) = exp(-iH_int t/ℏ).
    """
    H = np.kron(sigmaQ, sigmaE)
    U = scipy.linalg.expm(1j * H)
    return U

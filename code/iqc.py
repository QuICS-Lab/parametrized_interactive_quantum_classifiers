from iqc_multitargets import iqc_de_multi_target, next_power_of_two
import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted



class IQC(BaseEstimator, ClassifierMixin):
    """Classificador IQC binário treinável por JAX/Optax ou PSO."""

    def __init__(
        self,
        model_name=None,
        method="optimizer",
        iqc = None,
        number_of_params = 10,
        max_steps=200,
        learning_rate=0.01,
        optimizer=None,
        n_particles=10,
        options_optimizer=None,
        verbose=False,
        random_state=40,
        N_e=2
    ):
        self.method = method
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_particles = n_particles
        self.options_optimizer = options_optimizer
        self.verbose = verbose
        self.random_state = random_state
        self.iqc = iqc  # Default IQC model; can be changed to iqc_britoetal if needed
        self.number_of_params = number_of_params
        self.model_name = model_name
        self.N_e = N_e

        if self.iqc == None:
            raise ValueError("IQC model must be provided. Use iqc_zhangetal or iqc_britoetal.")
        if self.model_name == None:
            raise ValueError("Model name must be provided. Use 'iqc_zhangetal' or 'iqc_britoetal'.")

    def model(self, x, params):
        """Funcionamento do IQC; subclasses podem sobrescrever somente este método."""
        if self.model_name == "iqc_zhangetal":
            return self.iqc(x, np.array([1, 1, 1, 1]), params)  # Default alpha values; can be changed if needed
        elif self.model_name == "iqc_britoetal":
            return self.iqc(x, np.array([1, 1, 1, 1]), params)  # Default alpha values; can be changed if needed
        elif self.model_name == "iqc_alfa":
            return self.iqc(x, params[0:4], params[4:])  # Default alpha values; can be changed if needed
        elif self.model_name == "iqc_multidimensional_2":
            return self.iqc(x, params[0:4],params[4:], N_e=self.N_e)
        elif self.model_name == "iqc_multidimensional_4":
            return self.iqc(x, params[0:4],params[4:], N_e=self.N_e)
        elif self.model_name == "iqc_multidimensional_8":
            return self.iqc(x, params[0:4],params[4:], N_e=self.N_e)
        elif self.model_name == "iqc_de":
            return self.iqc(x, params[0:4],params[4:], N_e=self.N_e)
        elif self.model_name == "iqc_de_multitarget_2":
            return self.iqc(x, params[0:2*4],params[2*4:], N_e=self.N_e,N_qubits_tgt=2)
        elif self.model_name == "iqc_de_multitarget_3":
            return self.iqc(x, params[0:3*4],params[3*4:], N_e=self.N_e,N_qubits_tgt=3)
        elif self.model_name == "iqc_multidimensional_2_multitarget_2":
            return self.iqc(x, params[0:2*4],params[2*4:], N_e=self.N_e,N_qubits_tgt=2)
        elif self.model_name == "iqc_multidimensional_2_multitarget_3":     
         return self.iqc(x, params[0:3*4],params[3*4:], N_e=self.N_e,N_qubits_tgt=3)
        elif self.model_name == "iqc_multidimensional_4_multitarget_2":
            return self.iqc(x, params[0:2*4],params[2*4:], N_e=self.N_e,N_qubits_tgt=2)
        elif self.model_name == "iqc_multidimensional_4_multitarget_3":
            return self.iqc(x, params[0:3*4],params[3*4:], N_e=self.N_e,N_qubits_tgt=3)
        elif self.model_name == "iqc_multidimensional_8_multitarget_2":
            return self.iqc(x, params[0:2*4],params[2*4:], N_e=self.N_e,N_qubits_tgt=2)
        elif self.model_name == "iqc_multidimensional_8_multitarget_3":
            return self.iqc(x, params[0:3*4],params[3*4:], N_e=self.N_e,N_qubits_tgt=3)       
        else:
            raise ValueError("Invalid model name. Use 'iqc_zhangetal' or 'iqc_britoetal'.")

    def _encode_y(self, y):
        return jnp.asarray(np.where(y == self.classes_[0], -1.0, 1.0))

    def _initial_params(self):
        key = jax.random.PRNGKey(self.random_state)
        return jax.random.uniform(
            key,
            shape=(self.number_of_params,),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )

    def _model_batch(self, params, X):
        return jax.vmap(lambda x: self.model(x,params))(X)

    def _loss(self, params, X, y):
        predictions = self._model_batch(params, X)
        return jnp.mean((predictions - y) ** 2)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("IQC suporta apenas classificação binária.")

        self.n_features_in_ = X.shape[1]

        if self.method == "optimizer":
            return self.fit_optimizer(X, y)
        if self.method == "pso":
            return self.fit_pso(X, y)

        raise ValueError("method deve ser 'optimizer' ou 'pso'.")

    def fit_optimizer(self, X, y):
        X = jnp.asarray(X)
        y = self._encode_y(y)
        params = self._initial_params()  # +1 for bias term

        optimizer = self.optimizer
        if optimizer is None:
            optimizer = optax.adam(self.learning_rate)

        opt_state = optimizer.init(params)

        """
        loss_and_grad = jax.jit(jax.value_and_grad(self._loss))

        for step in range(self.max_steps):
            loss, grads = loss_and_grad(params, X, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if self.verbose and step % 100 == 0:
                print(f"step={step}, loss={float(loss):.6f}")

        self.params_ = np.asarray(params)
        self.best_loss_ = float(self._loss(params, X, y))
        return self
        """
        def train_step(params, opt_state):
            loss, grads = jax.value_and_grad(
                self._loss
            )(params, X, y)

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
            )

            params = optax.apply_updates(
                params,
                updates,
            )

            return params, opt_state, loss


        train_step_jit = jax.jit(train_step)

        for step in range(self.max_steps):
            params, opt_state, loss = train_step_jit(
                params,
                opt_state,
            )
        self.params_ = np.asarray(params)
        self.best_loss_ = float(self._loss(params, X, y))
        return self

    def fit_pso(self, X, y):
        """
        try:
            from pyswarms.single import GlobalBestPSO
        except ImportError as exc:
            raise ImportError(
                "O treinamento por PSO requer a biblioteca pyswarms."
            ) from exc

        X_jax = jnp.asarray(X)
        y_jax = self._encode_y(y)

        def objective(particles):
            return np.asarray([
                float(self._loss(jnp.asarray(params), X_jax, y_jax))
                for params in particles
            ])

        
        options = self.options_optimizer
        if options is None:
            options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.number_of_params,  # +1 for bias term
            options=options,
        )
        for ITER in range(self.max_steps):
            best_loss, best_params = optimizer.optimize(
                            objective,
                            iters=1,
                            verbose=self.verbose,
            )
            if best_loss <= 0.001:
                print("finalizado na iteracao", ITER)
                break
            

        self.params_ = np.asarray(best_params)
        self.best_loss_ = float(best_loss)
        return self
        """
   
        try:
            from pyswarms.single import GlobalBestPSO
        except ImportError as exc:
            raise ImportError(
                "O treinamento por PSO requer a biblioteca pyswarms."
            ) from exc

        X_jax = jnp.asarray(X, dtype=jnp.float32)
        y_jax = self._encode_y(y)

        loss_jit = jax.jit(self._loss)

        # Compila antes de iniciar o PSO.
        initial_params = jnp.zeros(
            self.number_of_params,
            dtype=jnp.float32,
        )

        loss_jit(
            initial_params,
            X_jax,
            y_jax,
        ).block_until_ready()

        def objective(particles):
            return np.asarray([
                float(
                    loss_jit(
                        jnp.asarray(params, dtype=jnp.float32),
                        X_jax,
                        y_jax,
                    )
                )
                for params in particles
            ])

        options = self.options_optimizer
        if options is None:
            options = {
                "c1": 0.5,
                "c2": 0.3,
                "w": 0.9,
            }

        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.number_of_params,
            options=options,
        )

        for iteration in range(self.max_steps):
            best_loss, best_params = optimizer.optimize(
                objective,
                iters=1,
                verbose=self.verbose,
            )

            if best_loss <= 0.001:
                if self.verbose:
                    print("Finalizado na iteração", iteration)
                break

        self.params_ = np.asarray(best_params)
        self.best_loss_ = float(best_loss)

        return self

    def decision_function(self, X):
        check_is_fitted(self, ["params_", "classes_"])
        X = check_array(X)
        return np.asarray(
            self._model_batch(jnp.asarray(self.params_), jnp.asarray(X))
        )

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
# File: bayesian_symbolic_regression_finetuner.py
"""
Bayesian Symbolic Regression Fine-Tuner with Uncertainty Quantification.

Implements:
- Weighted Bayesian updating with old + new data
- One-time fine-tuning mode
- Dynamic segmented fine-tuning mode (x-fold)
- Laplace approximation for posterior estimation
- Hamiltonian Monte Carlo sampling
- Convergence criteria for dynamic updating
- Uncertainty propagation through sigmoid activation
"""

import numpy as np
import pandas as pd
import re
import math
import warnings
from scipy.optimize import least_squares, minimize
from scipy.stats import multivariate_normal
from sklearn.metrics import r2_score, f1_score, confusion_matrix
import copy
from typing import Dict, Tuple, List

warnings.filterwarnings('ignore')


class BayesianSymbolicRegressionFinetuner:
    """
    Bayesian fine-tuner for symbolic regression with uncertainty quantification.

    Features:
    - Weighted likelihood combining old and new data
    - Gaussian prior centered at original parameters
    - Laplace approximation or HMC for posterior estimation
    - One-time or dynamic (x-fold) updating
    - Convergence monitoring
    - Predictive uncertainty propagation
    """

    def __init__(self, equation_str, feature_names=None, temperature=5.0,
                 epsilon_threshold=0.05, use_numpy=False,
                 prior_scale=1.0, inference_method='laplace',
                 hmc_num_steps=20, hmc_step_size=0.01):
        """
        Parameters:
        -----------
        equation_str : str
            Symbolic equation from Site 1
        feature_names : list, optional
            Feature names
        temperature : float
            Initial temperature for sigmoid
        epsilon_threshold : float
            Classification threshold
        use_numpy : bool
            Use numpy functions
        prior_scale : float
            Prior variance scale: Σ₀ = prior_scale * I
            Larger = weaker prior (more data influence)
        inference_method : str
            'laplace' or 'hmc'
        hmc_num_steps : int
            Number of leapfrog steps in HMC
        hmc_step_size : float
            Step size for HMC integration
        """
        self.equation_str_original = equation_str
        self.equation_str = equation_str
        self.feature_names = feature_names
        self.use_numpy = use_numpy
        self.temperature_original = temperature
        self.temperature_current = temperature
        self.epsilon_threshold = epsilon_threshold

        self.prior_scale = prior_scale
        self.inference_method = inference_method
        self.hmc_num_steps = hmc_num_steps
        self.hmc_step_size = hmc_step_size

        if inference_method not in ['laplace', 'hmc']:
            raise ValueError(f"inference_method must be 'laplace' or 'hmc', got {inference_method}")

        self.constants_original = None
        self.constants_current = None
        self.n_constants = None

        # Posterior distribution tracking
        self.posterior_mean = None
        self.posterior_cov = None
        self.posterior_samples = None

        self._setup_math_functions()
        self._parse_equation()

    def _setup_math_functions(self):
        """Setup available math functions with safety wrappers."""

        def safe_log(x):
            if isinstance(x, np.ndarray):
                return np.where(x > 1e-15, np.log(x), np.nan)
            else:
                return math.log(x) if x > 1e-15 else float('nan')

        def safe_sqrt(x):
            if isinstance(x, np.ndarray):
                return np.where(x >= 0, np.sqrt(x), np.nan)
            else:
                return math.sqrt(x) if x >= 0 else float('nan')

        def safe_asin(x):
            if isinstance(x, np.ndarray):
                return np.where((x >= -1) & (x <= 1), np.arcsin(x), np.nan)
            else:
                return math.asin(x) if -1 <= x <= 1 else float('nan')

        def safe_acos(x):
            if isinstance(x, np.ndarray):
                return np.where((x >= -1) & (x <= 1), np.arccos(x), np.nan)
            else:
                return math.acos(x) if -1 <= x <= 1 else float('nan')

        self.math_functions = {
            'abs': abs, 'round': round, 'min': min, 'max': max, 'pow': pow,
            'float': float, 'int': int,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': safe_asin, 'acos': safe_acos, 'atan': math.atan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'exp': math.exp, 'log': safe_log, 'log10': math.log10, 'log2': math.log2,
            'sqrt': safe_sqrt, 'radians': math.radians, 'degrees': math.degrees,
            'pi': math.pi, 'e': math.e, 'tau': math.tau, 'inf': math.inf, 'nan': math.nan,
        }

        if self.use_numpy:
            np_functions = {
                'np_sin': np.sin, 'np_cos': np.cos, 'np_tan': np.tan,
                'np_exp': np.exp, 'np_log': lambda x: np.where(x > 1e-15, np.log(x), np.nan),
                'np_log10': np.log10, 'np_sqrt': lambda x: np.where(x >= 0, np.sqrt(x), np.nan),
                'np_power': np.power, 'np_arcsin': safe_asin, 'np_arccos': safe_acos,
                'np_arctan': np.arctan, 'np_sinh': np.sinh, 'np_cosh': np.cosh,
                'np_tanh': np.tanh, 'np_ceil': np.ceil, 'np_floor': np.floor,
            }
            self.math_functions.update(np_functions)

    def _parse_equation(self):
        """Extract constants from equation string."""
        equation = self.equation_str_original
        pattern = r'(?<![x\w])[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?(?![x\d])'

        matches = list(re.finditer(pattern, equation))

        if not matches:
            raise ValueError(f"No numeric constants found in equation: {equation}")

        self.constants_original = {}
        replacements = []

        for idx, match in enumerate(matches):
            const_value = float(match.group())
            self.constants_original[f'c{idx}'] = const_value
            replacements.append((match.group(), f'c{idx}', match.start(), match.end()))

        equation_with_c = equation
        for original_str, c_name, start, end in reversed(replacements):
            equation_with_c = equation_with_c[:start] + c_name + equation_with_c[end:]

        self.equation_str = equation_with_c
        self.n_constants = len(self.constants_original)
        self.constants_current = copy.deepcopy(self.constants_original)

        # Initialize posterior at prior (original values)
        self.posterior_mean = np.array(list(self.constants_original.values()) + [self.temperature_original])
        self.posterior_cov = np.eye(self.n_constants + 1) * self.prior_scale

        print(f"Parsed equation: {self.equation_str}")
        print(f"Extracted {self.n_constants} constants + 1 temperature = {self.n_constants + 1} parameters")
        print(f"Prior scale: {self.prior_scale}")
        print(f"Inference method: {self.inference_method}")

    def evaluate_expression(self, expression, features_dict):
        """Safely evaluate expression."""
        safe_dict = {**features_dict, **self.math_functions}

        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            if isinstance(result, complex):
                result = result.real
            result = float(result)
            if np.isnan(result) or np.isinf(result):
                return float('nan')
            return result
        except:
            return float('nan')

    def predict_single_raw(self, features_dict, constants_dict=None):
        """Raw prediction WITHOUT sigmoid activation."""
        if constants_dict is None:
            constants_dict = self.constants_original

        combined_dict = {**features_dict, **constants_dict}
        pred = self.evaluate_expression(self.equation_str, combined_dict)
        return pred

    def predict_batch_raw(self, X, constants_dict=None, feature_names=None):
        """Batch predictions WITHOUT sigmoid activation."""
        if feature_names is None:
            feature_names = self.feature_names or [f'x{i}' for i in range(X.shape[1])]

        predictions = []
        for sample in X:
            features_dict = {name: sample[j] for j, name in enumerate(feature_names)}
            pred = self.predict_single_raw(features_dict, constants_dict)
            if np.isnan(pred):
                pred = 0.0
            predictions.append(pred)

        return np.array(predictions)

    def sigmoid_activation(self, x, temperature=None):
        """Sigmoid activation with temperature parameter."""
        if temperature is None:
            temperature = self.temperature_current

        x = np.asarray(x)
        z = -x / temperature
        return 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))

    def sigmoid_derivative(self, x, temperature=None):
        """Derivative of sigmoid w.r.t. input."""
        if temperature is None:
            temperature = self.temperature_current

        s = self.sigmoid_activation(x, temperature)
        return s * (1 - s) / temperature

    def compute_dynamic_loss(self, y_true, y_pred_sigmoid):
        """Compute loosened loss for dynamic setting."""
        eps = self.epsilon_threshold
        loss = np.zeros_like(y_true, dtype=float)

        soil_mask = y_true <= eps
        loss[soil_mask] = np.maximum(0, y_pred_sigmoid[soil_mask] - eps) ** 2

        rock_mask = y_true >= (1 - eps)
        loss[rock_mask] = np.maximum(0, (1 - y_pred_sigmoid[rock_mask]) - eps) ** 2

        mixed_mask = ~soil_mask & ~rock_mask
        loss[mixed_mask] = (y_true[mixed_mask] - y_pred_sigmoid[mixed_mask]) ** 2

        return loss

    def compute_weighted_loss(self, X_old, y_old, X_new, y_new, theta):
        """
        Compute weighted combined loss from old and new data.

        Loss = w_old * sum(loss_old) + w_new * sum(loss_new)
        where w_old = w_new = 1 (equal total weight)
        """
        feature_names = self.feature_name or [f'x{i}' for i in range(X_old.shape[1])]

        # Extract constants and temperature from theta
        c_dict = {f'c{i}': theta[i] for i in range(self.n_constants)}
        temp = np.clip(theta[-1], 0.01, 100.0)

        # Predictions
        pred_raw_old = self.predict_batch_raw(X_old, c_dict, feature_names)
        pred_sigmoid_old = self.sigmoid_activation(pred_raw_old, temp)

        pred_raw_new = self.predict_batch_raw(X_new, c_dict, feature_names)
        pred_sigmoid_new = self.sigmoid_activation(pred_raw_new, temp)

        # Loss
        loss_old = self.compute_dynamic_loss(y_old, pred_sigmoid_old)
        loss_new = self.compute_dynamic_loss(y_new, pred_sigmoid_new)

        # Weighted combination (equal total weight)
        n_old = len(X_old)
        n_new = len(X_new)

        # Average loss per dataset
        avg_loss = (np.mean(loss_old) + np.mean(loss_new)) / 2.0

        return avg_loss, pred_sigmoid_old, pred_sigmoid_new

    def compute_log_posterior(self, theta, X_old, y_old, X_new, y_new):
        """
        Compute log posterior = log likelihood + log prior
        log p(θ | y) ∝ log p(y | θ) + log p(θ)
        """
        feature_names = self.feature_names or [f'x{i}' for i in range(X_old.shape[1])]

        # Ensure theta is valid
        theta = np.array(theta)
        theta[-1] = np.clip(theta[-1], 0.01, 100.0)

        c_dict = {f'c{i}': theta[i] for i in range(self.n_constants)}
        temp = theta[-1]

        # Predictions
        try:
            pred_raw_old = self.predict_batch_raw(X_old, c_dict, feature_names)
            pred_sigmoid_old = self.sigmoid_activation(pred_raw_old, temp)

            pred_raw_new = self.predict_batch_raw(X_new, c_dict, feature_names)
            pred_sigmoid_new = self.sigmoid_activation(pred_raw_new, temp)

            # Loss
            loss_old = self.compute_dynamic_loss(y_old, pred_sigmoid_old)
            loss_new = self.compute_dynamic_loss(y_new, pred_sigmoid_new)

            # Likelihood: -MSE
            n_old = len(X_old)
            n_new = len(X_new)

            # Weighted equally: each dataset contributes equally to total
            log_likelihood = -(np.mean(loss_old) + np.mean(loss_new)) / 2.0

            # Prior: Gaussian centered at original values
            theta_0 = np.concatenate([list(self.constants_original.values()), [self.temperature_original]])
            delta = theta - theta_0
            log_prior = -0.5 * np.sum(delta ** 2 / self.prior_scale)

            log_posterior = log_likelihood + log_prior

            return log_posterior

        except:
            return -1e10  # Return very negative value on error

    def compute_log_posterior_gradient(self, theta, X_old, y_old, X_new, y_new, epsilon=1e-5):
        """
        Compute gradient of log posterior using finite differences.
        ∇log p(θ) ≈ (log p(θ+ε) - log p(θ-ε)) / (2ε)
        """
        gradient = np.zeros_like(theta)

        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon

            log_post_plus = self.compute_log_posterior(theta_plus, X_old, y_old, X_new, y_new)
            log_post_minus = self.compute_log_posterior(theta_minus, X_old, y_old, X_new, y_new)

            gradient[i] = (log_post_plus - log_post_minus) / (2 * epsilon)

        return gradient

    def compute_log_posterior_hessian(self, theta, X_old, y_old, X_new, y_new, epsilon=1e-4):
        """
        Compute Hessian of negative log posterior using finite differences.
        H[i,j] ≈ (f(θ+ε_i+ε_j) - f(θ+ε_i) - f(θ+ε_j) + f(θ)) / ε²
        """
        n = len(theta)
        hessian = np.zeros((n, n))

        # Compute at central point
        f0 = self.compute_log_posterior(theta, X_old, y_old, X_new, y_new)

        # Compute gradient at central point
        grad0 = self.compute_log_posterior_gradient(theta, X_old, y_old, X_new, y_new, epsilon)

        for i in range(n):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon

            grad_plus = self.compute_log_posterior_gradient(theta_plus, X_old, y_old, X_new, y_new, epsilon)

            hessian[:, i] = (grad_plus - grad0) / epsilon

        # Make symmetric
        hessian = (hessian + hessian.T) / 2.0

        return -hessian  # Return negative (we want Hessian of negative log posterior)

    def laplace_approximation(self, X_old, y_old, X_new, y_new, verbose=True):
        """
        Laplace approximation: find MAP and approximate posterior as Gaussian.

        Returns:
        --------
        theta_map : posterior mean (MAP estimate)
        cov_posterior : posterior covariance (inverse Hessian)
        """
        if verbose:
            print("\n[Laplace Approximation]")
            print(f"  Data: n_old={len(X_old)}, n_new={len(X_new)}")

        feature_names = self.feature_names or [f'x{i}' for i in range(X_old.shape[1])]

        # Objective: negative log posterior
        def neg_log_posterior(theta):
            return -self.compute_log_posterior(theta, X_old, y_old, X_new, y_new)

        # Initial guess: current posterior mean
        theta_init = self.posterior_mean.copy()

        # Optimize
        result = minimize(
            neg_log_posterior,
            theta_init,
            method='BFGS',
            options={'maxiter': 1000, 'gtol': 1e-6},
        )

        theta_map = result.x
        theta_map[-1] = np.clip(theta_map[-1], 0.01, 100.0)

        if verbose:
            print(f"  MAP found: success={result.success}, nit={result.nit}")
            print(f"  MAP parameters: {theta_map}")

        # Compute Hessian at MAP
        hessian = self.compute_log_posterior_hessian(theta_map, X_old, y_old, X_new, y_new)

        # Posterior covariance = inverse Hessian
        try:
            cov_posterior = np.linalg.inv(hessian)
            # Ensure positive definite
            evals = np.linalg.eigvalsh(cov_posterior)
            if np.any(evals < 1e-10):
                if verbose:
                    print(f"  Warning: Hessian not positive definite. Adding regularization.")
                cov_posterior += np.eye(len(theta_map)) * 1e-4
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  Warning: Could not invert Hessian. Using identity.")
            cov_posterior = np.eye(len(theta_map)) * self.prior_scale

        if verbose:
            print(f"  Posterior cov (diagonal): {np.diag(cov_posterior)}")

        return theta_map, cov_posterior

    def hamiltonian_mc(self, X_old, y_old, X_new, y_new, num_samples=1000,
                       burn_in=200, verbose=True):
        """
        Hamiltonian Monte Carlo sampling from posterior.

        Returns:
        --------
        samples : posterior samples (num_samples - burn_in, n_params)
        """
        if verbose:
            print(f"\n[Hamiltonian MC]")
            print(f"  Sampling {num_samples} iterations (burn_in={burn_in})")

        n_params = self.n_constants + 1
        samples = []

        # Initialize at posterior mean
        theta = self.posterior_mean.copy()

        for iteration in range(num_samples):
            # Leapfrog integration
            momentum = np.random.randn(n_params)
            theta_prop = theta.copy()
            momentum_prop = momentum.copy()

            # Gradient at current position
            grad = self.compute_log_posterior_gradient(theta_prop, X_old, y_old, X_new, y_new)

            for step in range(self.hmc_num_steps):
                # Half step for momentum
                momentum_prop += 0.5 * self.hmc_step_size * grad

                # Full step for position
                theta_prop += self.hmc_step_size * momentum_prop
                theta_prop[-1] = np.clip(theta_prop[-1], 0.01, 100.0)

                # Gradient at new position
                grad = self.compute_log_posterior_gradient(theta_prop, X_old, y_old, X_new, y_new)

                # Half step for momentum
                momentum_prop += 0.5 * self.hmc_step_size * grad

            # Metropolis-Hastings acceptance
            log_post_current = self.compute_log_posterior(theta, X_old, y_old, X_new, y_new)
            log_post_prop = self.compute_log_posterior(theta_prop, X_old, y_old, X_new, y_new)

            log_accept_ratio = log_post_prop - log_post_current

            if np.log(np.random.uniform()) < log_accept_ratio:
                theta = theta_prop
                if verbose and iteration % 200 == 0:
                    print(f"  Iteration {iteration}: accepted")
            else:
                if verbose and iteration % 200 == 0:
                    print(f"  Iteration {iteration}: rejected")

            if iteration >= burn_in:
                samples.append(theta.copy())

        samples = np.array(samples)

        # Compute posterior statistics
        posterior_mean = np.mean(samples, axis=0)
        posterior_cov = np.cov(samples.T)

        if verbose:
            print(f"  Posterior mean: {posterior_mean}")
            print(f"  Posterior std: {np.sqrt(np.diag(posterior_cov))}")

        return samples, posterior_mean, posterior_cov

    def bayesian_update(self, X_old, y_old, X_new, y_new, verbose=True):
        """
        Perform Bayesian posterior update.

        Returns:
        --------
        posterior_mean : posterior mean parameters
        posterior_cov : posterior covariance
        samples : posterior samples (if HMC) or None (if Laplace)
        """
        if self.inference_method == 'laplace':
            theta_map, cov_posterior = self.laplace_approximation(X_old, y_old, X_new, y_new, verbose)
            self.posterior_mean = theta_map
            self.posterior_cov = cov_posterior
            self.posterior_samples = None

            return theta_map, cov_posterior, None

        elif self.inference_method == 'hmc':
            samples, post_mean, post_cov = self.hamiltonian_mc(
                X_old, y_old, X_new, y_new, verbose=verbose
            )
            self.posterior_mean = post_mean
            self.posterior_cov = post_cov
            self.posterior_samples = samples

            return post_mean, post_cov, samples

    def predict_with_uncertainty(self, X_test, feature_names=None):
        """
        Make predictions with uncertainty using posterior distribution.

        For each test sample, compute:
        - Mean prediction: E[y | θ] ≈ ∫ sigmoid(f(x; θ)) p(θ | data) dθ
        - Predictive uncertainty: Var[y | θ]

        Using Monte Carlo integration if HMC samples available, else Laplace approximation.
        """
        if feature_names is None:
            feature_names = self.feature_names or [f'x{i}' for i in range(X_test.shape[1])]

        n_test = len(X_test)
        n_params = self.n_constants + 1

        # Get samples from posterior
        if self.posterior_samples is not None:
            # Use HMC samples
            samples = self.posterior_samples
            n_samples = len(samples)
        else:
            # Use Laplace approximation: sample from Gaussian
            n_samples = 1000
            samples = np.random.multivariate_normal(self.posterior_mean, self.posterior_cov, n_samples)

        # Predictions for each sample
        predictions_samples = np.zeros((n_samples, n_test))

        for s in range(n_samples):
            theta_s = samples[s]
            c_dict_s = {f'c{i}': theta_s[i] for i in range(self.n_constants)}
            temp_s = np.clip(theta_s[-1], 0.01, 100.0)

            pred_raw_s = self.predict_batch_raw(X_test, c_dict_s, feature_names)
            pred_sigmoid_s = self.sigmoid_activation(pred_raw_s, temp_s)
            predictions_samples[s, :] = pred_sigmoid_s

        # Compute posterior predictive mean and variance
        pred_mean = np.mean(predictions_samples, axis=0)
        pred_var = np.var(predictions_samples, axis=0)
        pred_std = np.sqrt(pred_var)

        return pred_mean, pred_std, predictions_samples

    def classify_prediction(self, y_pred):
        """Classify predictions into 3 classes."""
        eps = self.epsilon_threshold
        classification = np.zeros_like(y_pred, dtype=int)
        classification[y_pred <= eps] = 0
        classification[y_pred >= (1 - eps)] = 2
        classification[(y_pred > eps) & (y_pred < (1 - eps))] = 1
        return classification

    def compute_segment_metrics(self, y_true, y_pred_mean, y_pred_std=None):
        """Compute metrics for predictions with uncertainty."""
        valid_mask = np.isfinite(y_pred_mean)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred_mean[valid_mask]

        if len(y_true_valid) == 0:
            return {'r2': 0.0, 'rmse': np.inf, 'accuracy': 0.0, 'f1': 0.0}

        r2 = r2_score(y_true_valid, y_pred_valid)
        rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))

        # Classification metrics
        y_true_class = self.classify_prediction(y_true_valid)
        y_pred_class = self.classify_prediction(y_pred_valid)

        accuracy = np.mean(y_true_class == y_pred_class)

        try:
            f1 = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
        except:
            f1 = 0.0

        # Coverage probability (if uncertainty available)
        coverage = None
        if y_pred_std is not None:
            y_pred_std_valid = y_pred_std[valid_mask]
            # 95% credible interval
            lower = y_pred_valid - 1.96 * y_pred_std_valid
            upper = y_pred_valid + 1.96 * y_pred_std_valid
            coverage = np.mean((y_true_valid >= lower) & (y_true_valid <= upper))

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'accuracy': accuracy,
            'f1': f1,
            'coverage': coverage,
        }

        return metrics

    def bayesian_finetune_onetime(self, X_old, y_old, X_new, y_new,
                                  finetune_ratio=0.3, verbose=True):
        """
        One-time Bayesian fine-tuning using fixed portion of new data.

        Parameters:
        -----------
        X_old, y_old : old site data
        X_new, y_new : new site data (all available)
        finetune_ratio : fraction of new data to use for fine-tuning (default 30%)
        verbose : bool

        Returns:
        --------
        results : dict with posterior estimates and metrics
        """
        if verbose:
            print("\n" + "=" * 80)
            print("ONE-TIME BAYESIAN FINE-TUNING")
            print("=" * 80)
            print(f"Old data: {len(X_old)} samples")
            print(f"New data available: {len(X_new)} samples")
            print(f"Fine-tune ratio: {finetune_ratio}")

        # Select portion of new data for fine-tuning
        n_finetune = int(finetune_ratio * len(X_new))
        finetune_indices = np.random.choice(len(X_new), n_finetune, replace=False)

        X_new_finetune = X_new[finetune_indices]
        y_new_finetune = y_new[finetune_indices]

        # Remaining new data for evaluation
        eval_indices = np.array([i for i in range(len(X_new)) if i not in finetune_indices])
        X_new_eval = X_new[eval_indices]
        y_new_eval = y_new[eval_indices]

        if verbose:
            print(f"Fine-tune data: {len(X_new_finetune)} samples")
            print(f"Evaluation data: {len(X_new_eval)} samples")

        # Bayesian update
        post_mean, post_cov, samples = self.bayesian_update(
            X_old, y_old, X_new_finetune, y_new_finetune, verbose=verbose
        )

        # Evaluate on test set
        y_pred_mean, y_pred_std, _ = self.predict_with_uncertainty(X_new_eval)
        metrics = self.compute_segment_metrics(y_new_eval, y_pred_mean, y_pred_std)

        if verbose:
            print(f"\nEvaluation metrics:")
            print(f"  R²: {metrics['r2']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  Accuracy: {metrics['accuracy']:.6f}")
            print(f"  F1: {metrics['f1']:.6f}")
            if metrics['coverage'] is not None:
                print(f"  Coverage (95%): {metrics['coverage']:.4f}")

        results = {
            'mode': 'onetime',
            'posterior_mean': post_mean,
            'posterior_cov': post_cov,
            'posterior_samples': samples,
            'finetune_indices': finetune_indices,
            'eval_indices': eval_indices,
            'y_pred_mean': y_pred_mean,
            'y_pred_std': y_pred_std,
            'metrics': metrics,
        }

        return results

    def bayesian_finetune_dynamic(self, X_old, y_old, X_new, y_new,
                                  n_folds=5, convergence_tol=0.01,
                                  max_iterations=None, verbose=True):
        """
        Dynamic Bayesian fine-tuning with x-fold segments.

        Algorithm:
        1. Partition new data into n_folds segments
        2. For each fold k:
           a. Test segment k with posterior from k-1
           b. Accumulate segment k
           c. Update posterior using old + accumulated new data
           d. Check convergence

        Parameters:
        -----------
        X_old, y_old : old site data
        X_new, y_new : new site data
        n_folds : number of segments
        convergence_tol : threshold for parameter change relative norm
        max_iterations : max dynamic iterations (default: n_folds)
        verbose : bool

        Returns:
        --------
        results : comprehensive results with segment-wise analysis
        """
        if max_iterations is None:
            max_iterations = n_folds

        if verbose:
            print("\n" + "=" * 80)
            print("DYNAMIC BAYESIAN FINE-TUNING")
            print("=" * 80)
            print(f"Old data: {len(X_old)} samples")
            print(f"New data: {len(X_new)} samples")
            print(f"Number of folds: {n_folds}")
            print(f"Convergence tolerance: {convergence_tol}")

        feature_names = self.feature_names or [f'x{i}' for i in range(X_new.shape[1])]

        # Partition into folds
        fold_size = len(X_new) // n_folds
        folds = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(X_new)
            folds.append((start_idx, end_idx))

        if verbose:
            print(f"\nFold boundaries:")
            for i, (start, end) in enumerate(folds):
                print(f"  Fold {i}: [{start}:{end}] ({end - start} samples)")

        # Storage
        all_predictions_mean = np.zeros(len(X_new))
        all_predictions_std = np.zeros(len(X_new))
        fold_assignment = np.zeros(len(X_new), dtype=int)

        fold_metrics_list = []
        posterior_history = []
        convergence_history = []

        # Accumulated data
        X_accumulated = np.empty((0, X_new.shape[1]))
        y_accumulated = np.empty(0)

        if verbose:
            print(f"\n" + "-" * 80)
            print("STARTING ITERATIONS")
            print("-" * 80)

        for fold_idx, (fold_start, fold_end) in enumerate(folds):
            if verbose:
                print(f"\n[Fold {fold_idx}]")

            # Test data
            X_test = X_new[fold_start:fold_end]
            y_test = y_new[fold_start:fold_end]

            # Test with current posterior
            y_pred_mean, y_pred_std, _ = self.predict_with_uncertainty(X_test)
            all_predictions_mean[fold_start:fold_end] = y_pred_mean
            all_predictions_std[fold_start:fold_end] = y_pred_std
            fold_assignment[fold_start:fold_end] = fold_idx

            # Metrics
            fold_metrics = self.compute_segment_metrics(y_test, y_pred_mean, y_pred_std)
            fold_metrics_list.append(fold_metrics)

            if verbose:
                print(f"  Test metrics: R²={fold_metrics['r2']:.6f}, RMSE={fold_metrics['rmse']:.6f}")
                if fold_metrics['coverage'] is not None:
                    print(f"  Coverage: {fold_metrics['coverage']:.4f}")

            # Store posterior before update
            posterior_history.append({
                'fold': fold_idx,
                'phase': 'before_update',
                'posterior_mean': self.posterior_mean.copy(),
                'posterior_cov': self.posterior_cov.copy(),
            })

            # Check convergence
            if fold_idx > 0:
                prev_mean = posterior_history[-2]['posterior_mean']
                param_change = np.linalg.norm(self.posterior_mean - prev_mean) / (np.linalg.norm(prev_mean) + 1e-10)
                convergence_history.append({
                    'fold': fold_idx,
                    'parameter_change': param_change,
                    'converged': param_change < convergence_tol,
                })

                if verbose:
                    print(f"  Parameter change: {param_change:.6f} (tol: {convergence_tol})")

                if param_change < convergence_tol:
                    if verbose:
                        print(f"  Convergence achieved!")
                    # Can optionally stop here

            # Accumulate test data
            X_accumulated = np.vstack([X_accumulated, X_test])
            y_accumulated = np.concatenate([y_accumulated, y_test])

            if verbose:
                print(f"  Accumulated: {len(X_accumulated)} samples")

            # Update posterior (unless last fold)
            if fold_idx < len(folds) - 1 and fold_idx < max_iterations - 1:
                if verbose:
                    print(f"  Updating posterior...")

                post_mean, post_cov, samples = self.bayesian_update(
                    X_old, y_old, X_accumulated, y_accumulated, verbose=False
                )

                # Store updated posterior
                posterior_history.append({
                    'fold': fold_idx,
                    'phase': 'after_update',
                    'posterior_mean': post_mean.copy(),
                    'posterior_cov': post_cov.copy(),
                })

        # Final metrics on all test data
        final_metrics = self.compute_segment_metrics(y_new, all_predictions_mean, all_predictions_std)

        if verbose:
            print(f"\n" + "=" * 80)
            print("FINAL RESULTS")
            print("=" * 80)
            print(f"Overall R²: {final_metrics['r2']:.6f}")
            print(f"Overall RMSE: {final_metrics['rmse']:.6f}")
            print(f"Overall Accuracy: {final_metrics['accuracy']:.6f}")
            print(f"Overall F1: {final_metrics['f1']:.6f}")
            if final_metrics['coverage'] is not None:
                print(f"Overall Coverage: {final_metrics['coverage']:.4f}")

        results = {
            'mode': 'dynamic',
            'n_folds': n_folds,
            'posterior_mean': self.posterior_mean.copy(),
            'posterior_cov': self.posterior_cov.copy(),
            'posterior_samples': self.posterior_samples,
            'y_pred_mean': all_predictions_mean,
            'y_pred_std': all_predictions_std,
            'fold_assignment': fold_assignment,
            'fold_metrics': fold_metrics_list,
            'posterior_history': posterior_history,
            'convergence_history': convergence_history,
            'final_metrics': final_metrics,
        }

        return results

    def print_detailed_report(self, results):
        """Print comprehensive report of Bayesian fine-tuning results."""
        print("\n" + "=" * 80)
        print("BAYESIAN SYMBOLIC REGRESSION - DETAILED REPORT")
        print("=" * 80)

        print(f"\nInference method: {self.inference_method}")
        print(f"Prior scale: {self.prior_scale}")

        if results['mode'] == 'onetime':
            print(f"\nMode: ONE-TIME FINE-TUNING")
            print(f"  Posterior mean: {results['posterior_mean']}")
            print(f"  Posterior std: {np.sqrt(np.diag(results['posterior_cov']))}")
            print(f"\nMetrics:")
            for key, val in results['metrics'].items():
                if val is not None:
                    print(f"  {key}: {val:.6f}")

        elif results['mode'] == 'dynamic':
            print(f"\nMode: DYNAMIC FINE-TUNING")
            print(f"  Number of folds: {results['n_folds']}")
            print(f"  Posterior mean: {results['posterior_mean']}")
            print(f"  Posterior std: {np.sqrt(np.diag(results['posterior_cov']))}")

            print(f"\nPer-fold metrics:")
            for i, metrics in enumerate(results['fold_metrics']):
                print(f"  Fold {i}:")
                for key, val in metrics.items():
                    if val is not None:
                        print(f"    {key}: {val:.6f}")

            print(f"\nConvergence history:")
            for entry in results['convergence_history']:
                print(f"  Fold {entry['fold']}: param_change={entry['parameter_change']:.6f}, " +
                      f"converged={entry['converged']}")

            print(f"\nFinal metrics:")
            for key, val in results['final_metrics'].items():
                if val is not None:
                    print(f"  {key}: {val:.6f}")

        print(f"\n" + "=" * 80)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def demonstrate_bayesian_finetuning():
    """
    Example: One-time and dynamic Bayesian fine-tuning.
    """
    print("\n" + "=" * 80)
    print("BAYESIAN SYMBOLIC REGRESSION FINE-TUNING - DEMONSTRATION")
    print("=" * 80)

    # Load data
    df_site1 = pd.read_csv('E:/LIU Yifei/PySR/X_vali/datasets/prior_TBM1_clean_dimensionless_intv5+10.csv')
    X_site1 = df_site1.iloc[:, 1:].values
    y_site1 = df_site1.iloc[:, 0].values

    df_site2 = pd.read_csv('E:/LIU Yifei/PySR/X_vali/datasets/prior_TBM5_clean_dimensionless_intv5+10.csv')
    X_site2 = df_site2.iloc[:, 1:].values
    y_site2 = df_site2.iloc[:, 0].values

    print(f"\nSite 1: {len(X_site1)} samples")
    print(f"Site 2: {len(X_site2)} samples")

    # Equation from Site 1
    equation = "sin(sin(((cos(x3) / (x5 + 0.28165534)) + sin(exp(x6 ** 2.8740988))) / ((1.2227962 ** x2) - 1.1422936)))"

    # Initialize Bayesian finetuner
    print("\n[1/3] Initializing Bayesian finetuner...")
    finetuner = BayesianSymbolicRegressionFinetuner(
        equation_str=equation,
        temperature=5.0,
        epsilon_threshold=0.05,
        prior_scale=1.0,
        inference_method='laplace',  # or 'hmc'
    )

    # One-time fine-tuning
    print("\n[2/3] ONE-TIME FINE-TUNING...")
    results_onetime = finetuner.bayesian_finetune_onetime(
        X_site1, y_site1,
        X_site2, y_site2,
        finetune_ratio=0.3,
        verbose=True
    )

    finetuner.print_detailed_report(results_onetime)

    # Dynamic fine-tuning
    print("\n[3/3] DYNAMIC FINE-TUNING...")
    results_dynamic = finetuner.bayesian_finetune_dynamic(
        X_site1, y_site1,
        X_site2, y_site2,
        n_folds=5,
        convergence_tol=0.01,
        verbose=True
    )

    finetuner.print_detailed_report(results_dynamic)

    # Save results
    output_dir = 'E:/LIU Yifei/PySR/X_vali/bayesian_results/'

    # Save one-time results
    onetime_df = pd.DataFrame({
        'y_true': results_onetime['y_pred_mean'],  # Use test data
        'y_pred_mean': results_onetime['y_pred_mean'],
        'y_pred_std': results_onetime['y_pred_std'],
    })
    onetime_df.to_csv(output_dir + 'bayesian_onetime.csv', index=False)

    # Save dynamic results
    dynamic_df = pd.DataFrame({
        'y_pred_mean': results_dynamic['y_pred_mean'],
        'y_pred_std': results_dynamic['y_pred_std'],
        'fold_assignment': results_dynamic['fold_assignment'],
    })
    dynamic_df.to_csv(output_dir + 'bayesian_dynamic.csv', index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    demonstrate_bayesian_finetuning()
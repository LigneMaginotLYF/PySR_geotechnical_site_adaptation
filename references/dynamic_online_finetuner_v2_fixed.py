# File: dynamic_online_finetuner_v2_fixed.py
"""
Dynamic Online Learning Fine-Tuner for Symbolic Regression (v2).
Progressively fine-tunes formula as new data arrives in ordered segments.

New Features (v2):
- Temperature parameter is now tunable (learned during fine-tuning)
- Differential scaling: separate lambda for formula constants vs temperature
- Cumulative vs reset mode: choose between progressive formula updates or reset per segment
- CRITICAL FIX: Temperature is properly carried forward between iterations in cumulative mode
- Full backward compatibility with v1
"""

import numpy as np
import pandas as pd
import re
import math
import warnings
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, f1_score, confusion_matrix
import copy

warnings.filterwarnings('ignore')


class DynamicOnlineFineTuner:
    """
    Dynamic online learning: fine-tune formula progressively on ordered data segments.

    Key features:
    - Sigmoid activation: s(x) = 1/(1+exp(-x/temperature))
    - Classification thresholds: epsilon (not 0/1)
    - Loosened loss: tolerates violations at boundaries
    - Progressive fitting: each segment tested on previously fine-tuned formula
    - TUNABLE TEMPERATURE: learned during optimization
    - DIFFERENTIAL SCALING: different regularization for constants vs temperature
    - CUMULATIVE/RESET MODES: choice of formula state between segments
    - TEMPERATURE CARRY-FORWARD: temperature properly maintained in cumulative mode
    """

    def __init__(self, equation_str, feature_names=None, temperature=1.0,
                 epsilon_threshold=0.05, use_numpy=False,
                 finetune_temperature=True, accumulation_mode='cumulative'):
        """
        Parameters:
        -----------
        equation_str : str
            Symbolic equation from original site (Site 1)
        feature_names : list, optional
            Feature names, e.g., ['x0', 'x1', 'x2', ...]
        temperature : float
            Initial temperature parameter for sigmoid activation (default: 1.0)
        epsilon_threshold : float
            Classification threshold for soil/rock vs mixed (default: 0.05)
        use_numpy : bool
            Whether to use numpy functions
        finetune_temperature : bool
            Whether to optimize temperature as a learnable parameter (default: True)
        accumulation_mode : str
            'cumulative': accumulate data and formulae across segments (progressive)
            'reset': reset formula to original before each segment (independent segments)
            (default: 'cumulative')
        """
        self.equation_str_original = equation_str
        self.equation_str = equation_str
        self.feature_names = feature_names
        self.use_numpy = use_numpy
        self.temperature_original = temperature
        self.temperature_current = temperature
        self.epsilon_threshold = epsilon_threshold

        self.finetune_temperature = finetune_temperature
        self.accumulation_mode = accumulation_mode

        if accumulation_mode not in ['cumulative', 'reset']:
            raise ValueError(f"accumulation_mode must be 'cumulative' or 'reset', got {accumulation_mode}")

        self.constants_original = None
        self.constants_current = None
        self.n_constants = None

        self._setup_math_functions()
        self._parse_equation()

        # Track history across iterations
        self.constants_history = []
        self.temperature_history = []
        self.iteration_metrics_history = []

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

        print(f"Parsed equation: {self.equation_str}")
        print(f"Extracted {self.n_constants} constants (initial values):")
        for k, v in self.constants_original.items():
            print(f"  {k} = {v:.6e}")
        print(f"\nInitial sigmoid temperature: {self.temperature_original}")
        print(f"Finetune temperature: {self.finetune_temperature}")
        print(f"Classification epsilon: {self.epsilon_threshold}")
        print(f"Accumulation mode: {self.accumulation_mode}")

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

        except (ValueError, ZeroDivisionError, OverflowError):
            return float('nan')
        except Exception:
            return float('nan')

    def predict_single_raw(self, features_dict, constants_dict=None):
        """
        Raw prediction WITHOUT sigmoid activation.
        """
        if constants_dict is None:
            constants_dict = self.constants_original

        combined_dict = {**features_dict, **constants_dict}
        pred = self.evaluate_expression(self.equation_str, combined_dict)

        return pred

    def predict_batch_raw(self, X, constants_dict=None, feature_names=None):
        """
        Batch predictions WITHOUT sigmoid activation.
        """
        if feature_names is None:
            feature_names = self.feature_names or [f'x{i}' for i in range(X.shape[1])]

        predictions = []
        nan_count = 0

        for sample in X:
            features_dict = {name: sample[j] for j, name in enumerate(feature_names)}
            pred = self.predict_single_raw(features_dict, constants_dict)

            if np.isnan(pred):
                nan_count += 1
                pred = 0.0  # Fallback for raw (will be transformed by sigmoid)

            predictions.append(pred)

        predictions = np.array(predictions)

        return predictions

    def sigmoid_activation(self, x, temperature=None):
        """
        Sigmoid activation with temperature parameter.
        s(x) = 1 / (1 + exp(-x/temperature))

        Parameters:
        -----------
        x : array-like
            Input values
        temperature : float, optional
            Temperature parameter. If None, uses current temperature.
        """
        if temperature is None:
            temperature = self.temperature_current

        # Avoid overflow
        x = np.asarray(x)
        z = -x / temperature
        return 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))

    def predict_batch_with_sigmoid(self, X, constants_dict=None, feature_names=None,
                                   temperature=None):
        """
        Batch predictions WITH sigmoid activation applied.
        Output is always in (0, 1).
        """
        pred_raw = self.predict_batch_raw(X, constants_dict, feature_names)
        return self.sigmoid_activation(pred_raw, temperature)

    def compute_dynamic_loss(self, y_true, y_pred_sigmoid):
        """
        Compute loosened loss for dynamic setting.

        Logic:
        - Soil (GT ≤ eps): penalize max(0, pred - eps)^2 only (tolerate pred < eps)
        - Rock (GT ≥ 1-eps): penalize max(0, (1-pred) - eps)^2 only (tolerate pred > 1-eps)
        - Mixed (eps < GT < 1-eps): standard MSE

        Parameters:
        -----------
        y_true : np.ndarray
            Ground truth values
        y_pred_sigmoid : np.ndarray
            Predictions after sigmoid (values in (0, 1))

        Returns:
        --------
        np.ndarray : Element-wise loss values
        """
        eps = self.epsilon_threshold
        loss = np.zeros_like(y_true, dtype=float)

        # Soil region (GT ≤ eps)
        soil_mask = y_true <= eps
        loss[soil_mask] = np.maximum(0, y_pred_sigmoid[soil_mask] - eps) ** 2

        # Rock region (GT ≥ 1-eps)
        rock_mask = y_true >= (1 - eps)
        loss[rock_mask] = np.maximum(0, (1 - y_pred_sigmoid[rock_mask]) - eps) ** 2

        # Mixed region (eps < GT < 1-eps)
        mixed_mask = ~soil_mask & ~rock_mask
        loss[mixed_mask] = (y_true[mixed_mask] - y_pred_sigmoid[mixed_mask]) ** 2

        return loss

    def classify_prediction(self, y_pred):
        """
        Classify predictions into 3 classes based on epsilon threshold.

        Returns:
        --------
        np.ndarray : Classification [0=soil, 1=mixed, 2=rock]
        """
        eps = self.epsilon_threshold
        classification = np.zeros_like(y_pred, dtype=int)
        classification[y_pred <= eps] = 0  # Soil
        classification[y_pred >= (1 - eps)] = 2  # Rock
        classification[(y_pred > eps) & (y_pred < (1 - eps))] = 1  # Mixed
        return classification

    def finetune_one_iteration(self, X_train, y_train, lambda_regularization=0.1,
                               lambda_temperature=None, verbose=False):
        """
        Fine-tune constants AND temperature on a single iteration using Regularized Least Squares.

        Minimizes: ||L(y, sigmoid(f(X; c), t))||^2 + λ_c * ||c - c_original||^2 + λ_t * ||t - t_original||^2

        Parameters:
        -----------
        X_train : np.ndarray
            Accumulated training data
        y_train : np.ndarray
            Accumulated targets
        lambda_regularization : float
            Regularization strength for formula constants (default: 0.1)
        lambda_temperature : float, optional
            Regularization strength for temperature. If None, uses same as lambda_regularization.
        verbose : bool
            Print optimization details

        Returns:
        --------
        dict : Optimization result with fitted constants, temperature, and metrics
        """
        if lambda_temperature is None:
            lambda_temperature = lambda_regularization

        feature_names = self.feature_names or [f'x{i}' for i in range(X_train.shape[1])]

        # Initial values (warm-start from current constants and temperature)
        c_initial = np.array([self.constants_current[f'c{i}'] for i in range(self.n_constants)])
        c_original = np.array([self.constants_original[f'c{i}'] for i in range(self.n_constants)])

        # Add temperature to optimization if enabled
        if self.finetune_temperature:
            theta_initial = np.concatenate([c_initial, [self.temperature_current]])
            theta_original = np.concatenate([c_original, [self.temperature_original]])
        else:
            theta_initial = c_initial
            theta_original = c_original

        def residual_with_regularization(theta_values):
            """
            Residual vector with dynamic loss + regularization.
            theta_values = [c0, c1, ..., c_n, temperature] (if finetune_temperature=True)
            or [c0, c1, ..., c_n] (if finetune_temperature=False)
            """
            if self.finetune_temperature:
                c_values = theta_values[:-1]
                temp_value = theta_values[-1]
                # Ensure temperature stays positive and bounded
                temp_value = np.clip(temp_value, 0.01, 100.0)
            else:
                c_values = theta_values
                temp_value = self.temperature_current

            constants_dict = {f'c{i}': c_values[i] for i in range(len(c_values))}

            # Get raw predictions
            pred_raw = self.predict_batch_raw(X_train, constants_dict, feature_names)

            # Apply sigmoid with current temperature
            pred_sigmoid = self.sigmoid_activation(pred_raw, temp_value)

            # Compute dynamic loss
            element_losses = self.compute_dynamic_loss(y_train, pred_sigmoid)

            # Take square root for least_squares
            data_residual = np.sqrt(np.abs(element_losses))

            # Regularization: L2 penalty toward ORIGINAL values
            reg_weight_c = np.sqrt(lambda_regularization)
            reg_term_c = reg_weight_c * (c_values - c_original)

            if self.finetune_temperature:
                reg_weight_t = np.sqrt(lambda_temperature)
                reg_term_t = reg_weight_t * (temp_value - self.temperature_original)
                reg_term = np.concatenate([reg_term_c, [reg_term_t]])
            else:
                reg_term = reg_term_c

            # Combine residuals
            combined_residuals = np.concatenate([data_residual, reg_term])

            return combined_residuals

        # Optimize
        if self.finetune_temperature:
            bounds = (np.concatenate([np.full(self.n_constants, -np.inf), [0.01]]),
                      np.concatenate([np.full(self.n_constants, np.inf), [100.0]]))
        else:
            bounds = (-np.inf, np.inf)

        result_opt = least_squares(
            residual_with_regularization,
            theta_initial,
            bounds=bounds,
            max_nfev=10000,
            verbose=1 if verbose else 0,
        )

        theta_fitted = result_opt.x

        # Extract fitted values
        if self.finetune_temperature:
            c_fitted = theta_fitted[:-1]
            self.temperature_current = np.clip(theta_fitted[-1], 0.01, 100.0)
        else:
            c_fitted = theta_fitted

        # Update current constants
        for i in range(self.n_constants):
            self.constants_current[f'c{i}'] = c_fitted[i]

        # Compute training metrics
        pred_raw_train = self.predict_batch_raw(X_train, self.constants_current, feature_names)
        pred_sigmoid_train = self.sigmoid_activation(pred_raw_train, self.temperature_current)

        valid_mask = np.isfinite(pred_sigmoid_train)
        if np.sum(valid_mask) > 0:
            train_r2 = r2_score(y_train[valid_mask], pred_sigmoid_train[valid_mask])
            train_loss = np.mean(self.compute_dynamic_loss(y_train[valid_mask],
                                                           pred_sigmoid_train[valid_mask]))
        else:
            train_r2 = 0.0
            train_loss = np.inf

        return {
            'constants_fitted': copy.deepcopy(self.constants_current),
            'temperature_fitted': self.temperature_current,
            'train_r2': train_r2,
            'train_loss': train_loss,
            'n_iterations': result_opt.nfev,
            'success': result_opt.success,
        }

    def compute_segment_metrics(self, y_true, y_pred_sigmoid):
        """
        Compute comprehensive metrics for a test segment.

        Returns:
        --------
        dict : Metrics including R², F1 scores, accuracy, confusion matrix
        """
        # Filter valid predictions
        valid_mask = np.isfinite(y_pred_sigmoid)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred_sigmoid[valid_mask]

        if len(y_true_valid) == 0:
            return {
                'r2_overall': 0.0,
                'r2_mixed': np.nan,
                'rmse': np.inf,
                'class_accuracy': 0.0,
                'f1_weighted': 0.0,
                'f1_macro': 0.0,
                'f1_pure0': 0.0,
                'f1_mixed': 0.0,
                'f1_pure1': 0.0,
                'confusion_matrix': np.zeros((3, 3), dtype=int),
                'valid_count': 0,
            }

        # Overall R²
        r2_overall = r2_score(y_true_valid, y_pred_valid)

        # Mixed R² (only samples with mixed GT)
        eps = self.epsilon_threshold
        mixed_mask = (y_true_valid > eps) & (y_true_valid < (1 - eps))
        if np.sum(mixed_mask) > 0:
            r2_mixed = r2_score(y_true_valid[mixed_mask], y_pred_valid[mixed_mask])
        else:
            r2_mixed = np.nan

        # RMSE
        rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))

        # Classification metrics
        y_true_class = self.classify_prediction(y_true_valid)
        y_pred_class = self.classify_prediction(y_pred_valid)

        class_accuracy = np.mean(y_true_class == y_pred_class)

        try:
            f1_weighted = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
            f1_macro = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
            f1_per_class = f1_score(y_true_class, y_pred_class, average=None, zero_division=0)
        except:
            f1_weighted = f1_macro = 0.0
            f1_per_class = np.array([0.0, 0.0, 0.0])

        cm = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1, 2])

        metrics = {
            'r2_overall': r2_overall,
            'r2_mixed': r2_mixed,
            'rmse': rmse,
            'class_accuracy': class_accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_pure0': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
            'f1_mixed': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
            'f1_pure1': f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
            'confusion_matrix': cm,
            'valid_count': len(y_true_valid),
        }

        return metrics

    def finetune_dynamic_online(self, X_new, y_new, step_size=50,
                                lambda_regularization=0.1, lambda_temperature=None,
                                verbose=True):
        """
        Progressive fine-tuning on ordered data segments.

        **Algorithm** (Cumulative Mode):
        1. Partition X_new, y_new into segments of size step_size
        2. For each segment i:
           a. Test segment i using formula fine-tuned on segments [0:i]
           b. Accumulate segment i into training data
           c. Fine-tune formula (and temperature if enabled) on all accumulated data [0:i+1]
           d. IMPORTANT: Carry forward fitted temperature to next segment
        3. Track predictions from each segment + corresponding formula state

        **Algorithm** (Reset Mode):
        1. Partition X_new, y_new into segments of size step_size
        2. For each segment i:
           a. Reset constants and temperature to original values
           b. Test segment i using original formula
           c. Fine-tune formula on ONLY segment i (not accumulated)
           d. Reset for next segment (temperature reset to original)

        Parameters:
        -----------
        X_new : np.ndarray, shape (n_samples, n_features)
            Ordered data from new site (NOT SHUFFLED)
        y_new : np.ndarray, shape (n_samples,)
            Ordered targets from new site
        step_size : int
            Number of samples per segment (hyperparameter)
        lambda_regularization : float
            Regularization strength for formula constants
        lambda_temperature : float, optional
            Regularization strength for temperature. If None, uses same as lambda_regularization.
        verbose : bool
            Print progress and metrics

        Returns:
        --------
        dict : Comprehensive results with predictions, metrics, and constants history
        """
        if lambda_temperature is None:
            lambda_temperature = lambda_regularization

        if verbose:
            print("\n" + "=" * 80)
            print("DYNAMIC ONLINE FINE-TUNING (v2)")
            print("=" * 80)
            print(f"New site samples: {len(X_new)}")
            print(f"Step size (segment length): {step_size}")
            print(f"Number of segments: {int(np.ceil(len(X_new) / step_size))}")
            print(f"Lambda (constants): {lambda_regularization}")
            print(f"Lambda (temperature): {lambda_temperature}")
            print(f"Finetune temperature: {self.finetune_temperature}")
            print(f"Accumulation mode: {self.accumulation_mode}")

        feature_names = self.feature_names or [f'x{i}' for i in range(X_new.shape[1])]

        # Partition data into segments
        n_segments = int(np.ceil(len(X_new) / step_size))
        segments = []
        for i in range(n_segments):
            start_idx = i * step_size
            end_idx = min((i + 1) * step_size, len(X_new))
            segments.append((start_idx, end_idx))

        if verbose:
            print(f"\nSegment boundaries:")
            for i, (start, end) in enumerate(segments):
                print(f"  Segment {i}: samples [{start}:{end}] (size: {end - start})")

        # Storage for results
        all_predictions = np.zeros(len(X_new))
        all_predictions_class = np.zeros(len(X_new), dtype=int)
        segment_assignment = np.zeros(len(X_new), dtype=int)

        iteration_metrics_list = []
        constants_evolution = []
        temperature_evolution = []

        # Accumulated training data
        X_accumulated = np.empty((0, X_new.shape[1]))
        y_accumulated = np.empty(0)

        if verbose:
            print("\n" + "-" * 80)
            print("STARTING ITERATIONS")
            print("-" * 80)

        for iter_idx, (seg_start, seg_end) in enumerate(segments):
            if verbose:
                print(f"\n[Iteration {iter_idx}]")
                print(f"  Testing on segment [{seg_start}:{seg_end}]")
                print(f"  Mode: {self.accumulation_mode}")

            # Reset formula if in reset mode (BEFORE testing)
            if self.accumulation_mode == 'reset':
                self.constants_current = copy.deepcopy(self.constants_original)
                self.temperature_current = self.temperature_original
                X_accumulated = np.empty((0, X_new.shape[1]))
                y_accumulated = np.empty(0)
                if verbose:
                    print(f"  [RESET MODE] Formula reset to original")

            # Test current segment with previously fine-tuned formula
            X_test = X_new[seg_start:seg_end]
            y_test = y_new[seg_start:seg_end]

            pred_raw_test = self.predict_batch_raw(X_test, self.constants_current, feature_names)
            pred_sigmoid_test = self.sigmoid_activation(pred_raw_test, self.temperature_current)

            # Store predictions
            all_predictions[seg_start:seg_end] = pred_sigmoid_test
            all_predictions_class[seg_start:seg_end] = self.classify_prediction(pred_sigmoid_test)
            segment_assignment[seg_start:seg_end] = iter_idx

            # Compute metrics for this test segment
            seg_metrics = self.compute_segment_metrics(y_test, pred_sigmoid_test)
            iteration_metrics_list.append(seg_metrics)

            if verbose:
                print(f"  Test metrics:")
                print(f"    R² (overall): {seg_metrics['r2_overall']:.6f}")
                print(f"    R² (mixed):   {seg_metrics['r2_mixed']:.6f}")
                print(f"    RMSE:         {seg_metrics['rmse']:.6f}")
                print(f"    Accuracy:     {seg_metrics['class_accuracy']:.6f}")
                print(f"    F1 (weighted):{seg_metrics['f1_weighted']:.6f}")

            # Store constants at this state (before this iteration's fine-tuning)
            constants_evolution.append({
                'iteration': iter_idx,
                'phase': 'before_finetune',
                'constants': copy.deepcopy(self.constants_current),
            })

            temperature_evolution.append({
                'iteration': iter_idx,
                'phase': 'before_finetune',
                'temperature': self.temperature_current,
            })

            # Accumulate this segment into training data
            X_accumulated = np.vstack([X_accumulated, X_test])
            y_accumulated = np.concatenate([y_accumulated, y_test])

            if verbose:
                print(f"  Accumulated training data: {len(X_accumulated)} samples")

            # Fine-tune on accumulated data (unless this is the last segment)
            if iter_idx < len(segments) - 1:
                if verbose:
                    print(f"  Fine-tuning on accumulated data...")

                finetune_result = self.finetune_one_iteration(
                    X_accumulated, y_accumulated,
                    lambda_regularization=lambda_regularization,
                    lambda_temperature=lambda_temperature,
                    verbose=False
                )

                if verbose:
                    print(f"    Optimization success: {finetune_result['success']}")
                    print(f"    Training R²: {finetune_result['train_r2']:.6f}")
                    print(f"    Training loss: {finetune_result['train_loss']:.6f}")

                    if self.finetune_temperature:
                        temp_change = finetune_result['temperature_fitted'] - self.temperature_original
                        print(f"    Temperature: {finetune_result['temperature_fitted']:.6f} (Δ: {temp_change:+.6f})")

                    print(f"    Constant changes (first 3):\")")
                    for c_idx in range(min(3, self.n_constants)):
                        key = f'c{c_idx}'
                        old_val = self.constants_original[key]
                        new_val = self.constants_current[key]
                        change_pct = 100 * (new_val - old_val) / (abs(old_val) + 1e-10)
                        print(f"      {key}: {new_val:.6e} (Δ: {change_pct:+.2f}%)")
                    if self.n_constants > 3:
                        print(f"      ... ({self.n_constants - 3} more constants)")

                # Store constants after fine-tuning
                constants_evolution.append({
                    'iteration': iter_idx,
                    'phase': 'after_finetune',
                    'constants': copy.deepcopy(self.constants_current),
                })

                # CRITICAL: Store temperature after fine-tuning
                # This ensures temperature is carried forward to next segment in cumulative mode
                temperature_evolution.append({
                    'iteration': iter_idx,
                    'phase': 'after_finetune',
                    'temperature': self.temperature_current,
                })

        # Final evaluation on all samples
        if verbose:
            print("\n" + "=" * 80)
            print("FINAL EVALUATION ON ALL SAMPLES")
            print("=" * 80)

        final_metrics = self.compute_segment_metrics(y_new, all_predictions)

        if verbose:
            print(f"\nOverall metrics (all samples, using respective formulae):")
            print(f"  R² (overall):      {final_metrics['r2_overall']:.6f}")
            print(f"  R² (mixed):        {final_metrics['r2_mixed']:.6f}")
            print(f"  RMSE:              {final_metrics['rmse']:.6f}")
            print(f"  Class Accuracy:    {final_metrics['class_accuracy']:.6f}")
            print(f"  F1 (weighted):     {final_metrics['f1_weighted']:.6f}")
            print(f"  F1 (macro):        {final_metrics['f1_macro']:.6f}")
            print(f"\n  Per-class F1:")
            print(f"    Pure Soil (0):   {final_metrics['f1_pure0']:.6f}")
            print(f"    Mixed (1):       {final_metrics['f1_mixed']:.6f}")
            print(f"    Pure Rock (2):   {final_metrics['f1_pure1']:.6f}")
            print(f"\n  Confusion Matrix:")
            cm = final_metrics['confusion_matrix']
            print(f"              Pred 0  Pred Mixed  Pred 1")
            for i in range(3):
                row = cm[i]
                print(f"    True {i}:   {row[0]:6d}     {row[1]:6d}     {row[2]:6d}")

        # Prepare output
        result = {
            # Predictions and assignments
            'sample_predictions': all_predictions,
            'sample_predictions_class': all_predictions_class,
            'segment_assignment': segment_assignment,
            'y_true': y_new,

            # Per-iteration metrics
            'iteration_metrics': {f'iter_{i}': metrics for i, metrics in enumerate(iteration_metrics_list)},

            # Overall metrics
            'final_metrics': final_metrics,

            # Constants evolution
            'constants_history': constants_evolution,
            'constants_original': self.constants_original,
            'constants_final': self.constants_current,

            # Temperature evolution
            'temperature_history': temperature_evolution,
            'temperature_original': self.temperature_original,
            'temperature_final': self.temperature_current,

            # Metadata
            'step_size': step_size,
            'n_segments': n_segments,
            'lambda_regularization': lambda_regularization,
            'lambda_temperature': lambda_temperature,
            'finetune_temperature': self.finetune_temperature,
            'accumulation_mode': self.accumulation_mode,
            'epsilon_threshold': self.epsilon_threshold,
        }

        return result

    def print_detailed_report(self, results):
        """
        Print comprehensive report of dynamic online fine-tuning results.

        Parameters:
        -----------
        results : dict
            Output from finetune_dynamic_online()
        """
        print("\n" + "=" * 80)
        print("DYNAMIC ONLINE FINE-TUNING - DETAILED REPORT (v2)")
        print("=" * 80)

        # Configuration
        print(f"\nConfiguration:")
        print(f"  Temperature (original): {results['temperature_original']}")
        print(f"  Temperature (final):    {results['temperature_final']:.6f}")
        print(f"  Finetune temperature:   {results['finetune_temperature']}")
        print(f"  Epsilon threshold:      {results['epsilon_threshold']}")
        print(f"  Step size:              {results['step_size']}")
        print(f"  Lambda (constants):     {results['lambda_regularization']}")
        print(f"  Lambda (temperature):   {results['lambda_temperature']}")
        print(f"  Accumulation mode:      {results['accumulation_mode']}")
        print(f"  Number of segments:     {results['n_segments']}")

        # Iteration summary
        print(f"\n" + "-" * 80)
        print("PER-ITERATION METRICS")
        print("-" * 80)

        for i, (iter_key, metrics) in enumerate(results['iteration_metrics'].items()):
            print(f"\n{iter_key}:")
            print(f"  Samples tested: {metrics['valid_count']}")
            print(f"  R² (overall):   {metrics['r2_overall']:.6f}")
            print(f"  R² (mixed):     {metrics['r2_mixed']:.6f}")
            print(f"  RMSE:           {metrics['rmse']:.6f}")
            print(f"  Accuracy:       {metrics['class_accuracy']:.6f}")
            print(f"  F1 (weighted):  {metrics['f1_weighted']:.6f}")

        # Temperature evolution
        if results['finetune_temperature']:
            print(f"\n" + "-" * 80)
            print("TEMPERATURE EVOLUTION")
            print("-" * 80)

            for entry in results['temperature_history']:
                iter_num = entry['iteration']
                phase = entry['phase']
                temp = entry['temperature']
                delta = temp - results['temperature_original']
                change_pct = 100 * delta / results['temperature_original']
                print(f"\nIteration {iter_num} ({phase}):")
                print(f"  Temperature: {temp:.6f} (Δ: {delta:+.6f}, {change_pct:+.2f}%)")

        # Constants evolution (show only first 5)
        print(f"\n" + "-" * 80)
        print("CONSTANTS EVOLUTION (First 5 shown)")
        print("-" * 80)

        for entry in results['constants_history']:
            iter_num = entry['iteration']
            phase = entry['phase']
            constants = entry['constants']

            print(f"\nIteration {iter_num} ({phase}):")
            for c_key in sorted(constants.keys(), key=lambda x: int(x[1:]))[:5]:
                value = constants[c_key]
                original = results['constants_original'][c_key]
                change_pct = 100 * (value - original) / (abs(original) + 1e-10)
                print(f"  {c_key}: {value:.6e} (Δ: {change_pct:+.2f}%)")
            if len(constants) > 5:
                print(f"  ... ({len(constants) - 5} more constants)")

        # Final metrics
        print(f"\n" + "-" * 80)
        print("FINAL METRICS (ALL SAMPLES)")
        print("-" * 80)

        final = results['final_metrics']
        print(f"\nRegression Performance:")
        print(f"  Overall R²:      {final['r2_overall']:.6f}")
        print(f"  Mixed R²:        {final['r2_mixed']:.6f}")
        print(f"  RMSE:            {final['rmse']:.6f}")

        print(f"\nClassification Performance:")
        print(f"  Accuracy:        {final['class_accuracy']:.6f}")
        print(f"  F1 (weighted):   {final['f1_weighted']:.6f}")
        print(f"  F1 (macro):      {final['f1_macro']:.6f}")
        print(f"  F1 Pure Soil:    {final['f1_pure0']:.6f}")
        print(f"  F1 Mixed:        {final['f1_mixed']:.6f}")
        print(f"  F1 Pure Rock:    {final['f1_pure1']:.6f}")

        print(f"\nConfusion Matrix:")
        cm = final['confusion_matrix']
        print(f"           Pred 0  Pred Mixed  Pred 1")
        for i in range(3):
            print(f"  True {i}:  {cm[i, 0]:6d}     {cm[i, 1]:6d}     {cm[i, 2]:6d}")

        print(f"\n" + "=" * 80)


# ============================================================================
# USAGE EXAMPLE & HELPER FUNCTIONS
# ============================================================================

def load_data(filepath):
    """
    Load data from CSV file.
    Assumes first column is target, remaining columns are features.
    """
    df = pd.read_csv(filepath)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    return X, y


def demonstrate_dynamic_online_learning():
    """
    Complete example: Load data and run dynamic online fine-tuning.
    Shows cumulative mode with temperature tuning.
    """
    print("\n" + "=" * 80)
    print("DYNAMIC ONLINE FINE-TUNING (v2) - DEMONSTRATION")
    print("=" * 80)

    # ========== CONFIGURATION ==========
    SITE1_FILE = 'E:/LIU Yifei/PySR/X_vali/datasets/prior_TBM1_clean_dimensionless_intv5+10.csv'
    SITE2_FILE = 'E:/LIU Yifei/PySR/X_vali/datasets/prior_TBM5_clean_dimensionless_intv5+10.csv'

    # Equation from Site 1
    EQUATION_SITE1 = "sin(sin(((cos(x3) / (x5 + 0.20481215)) + sin(exp((x6 + 0.101746835) * x6))) / ((1.2297258 ** x2) - 1.1874241)) / cos(cos(x11 - x5))) + 0.37291053"

    FEATURE_NAMES = None

    # Hyperparameters
    TEMPERATURE_INIT = 0.2
    EPSILON_THRESHOLD = 0.05
    STEP_SIZE = 100
    LAMBDA_CONST = 0
    LAMBDA_TEMP = 0  # Different scaling for temperature
    FINETUNE_TEMPERATURE = True
    ACCUMULATION_MODE = 'reset'  # 'cumulative' or 'reset'

    # ========== LOAD DATA ==========
    print("\n[1/5] Loading data...")
    X_site1, y_site1 = load_data(SITE1_FILE)
    X_site2, y_site2 = load_data(SITE2_FILE)

    print(f"  Site 1 (original training): {len(X_site1)} samples")
    print(f"  Site 2 (new site, ordered): {len(X_site2)} samples")

    # ========== INITIALIZE FINETUNER ==========
    print("\n[2/5] Initializing dynamic online fine-tuner (v2)...")
    finetuner = DynamicOnlineFineTuner(
        equation_str=EQUATION_SITE1,
        feature_names=FEATURE_NAMES,
        temperature=TEMPERATURE_INIT,
        epsilon_threshold=EPSILON_THRESHOLD,
        finetune_temperature=FINETUNE_TEMPERATURE,
        accumulation_mode=ACCUMULATION_MODE,
    )

    # ========== RUN DYNAMIC ONLINE LEARNING ==========
    print("\n[3/5] Running dynamic online fine-tuning...")
    results = finetuner.finetune_dynamic_online(
        X_new=X_site2,
        y_new=y_site2,
        step_size=STEP_SIZE,
        lambda_regularization=LAMBDA_CONST,
        lambda_temperature=LAMBDA_TEMP,
        verbose=True
    )

    # ========== PRINT DETAILED REPORT ==========
    print("\n[4/5] Generating detailed report...")
    finetuner.print_detailed_report(results)

    # ========== SAVE RESULTS ==========
    print("\n[5/5] Saving results...")
    print("-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    # Save predictions
    results_df = pd.DataFrame({
        'y_true': results['y_true'],
        'y_pred': results['sample_predictions'],
        'y_pred_class': results['sample_predictions_class'],
        'segment_id': results['segment_assignment'],
    })

    output_file = 'E:/LIU Yifei/PySR/X_vali/dyn_results/dynamic_online_predictions_inctemp_reset_temp0.2.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Saved predictions to: {output_file}")

    # Save parameters
    params_file = output_file.replace('.csv', '_params.txt')
    with open(params_file, 'w') as f:
        f.write("Dynamic Online Fine-Tuning (v2) - Parameters\\n")
        f.write("=" * 80 + "\\n")
        f.write(f"Temperature (init): {results['temperature_original']}\\n")
        f.write(f"Temperature (final): {results['temperature_final']:.6f}\\n")
        f.write(f"Finetune temperature: {results['finetune_temperature']}\\n")
        f.write(f"Lambda (constants): {results['lambda_regularization']}\\n")
        f.write(f"Lambda (temperature): {results['lambda_temperature']}\\n")
        f.write(f"Accumulation mode: {results['accumulation_mode']}\\n")
        f.write(f"Step size: {results['step_size']}\\n")
        f.write(f"\\nFinal Metrics:\\n")
        f.write(f"  R² (overall): {results['final_metrics']['r2_overall']:.6f}\\n")
        f.write(f"  R² (mixed): {results['final_metrics']['r2_mixed']:.6f}\\n")
        f.write(f"  Accuracy: {results['final_metrics']['class_accuracy']:.6f}\\n")
    print(f"Saved parameters to: {params_file}")

    return finetuner, results


if __name__ == "__main__":
    finetuner, results = demonstrate_dynamic_online_learning()
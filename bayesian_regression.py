"""
bayesian_regression.py
----------------------

Stage 5 of the Symbolic Transfer Learning workflow.

Responsibilities
----------------
1. Build weighted design matrix.
2. Bayesian Linear Regression.
3. Compute posterior mean/covariance.
4. Return Bayesian model.

"""

import numpy as np


class BayesianLinearRegression:

    def __init__(
        self,
        alpha=1.0,
        beta=1.0
    ):

        self.alpha = alpha
        self.beta = beta

        self.posterior_mean = None
        self.posterior_cov = None

    # ---------------------------------------------------------

    @staticmethod
    def build_design_matrix(
        Phi_old,
        Phi_new,
        selected_indices,
        adapt_indices
    ):

        Phi_old = Phi_old[:, selected_indices]

        Phi_adapt = Phi_new[
            adapt_indices
        ][:, selected_indices]

        return Phi_old, Phi_adapt

    # ---------------------------------------------------------

    @staticmethod
    def build_targets(
        Y_old,
        Y_new,
        adapt_indices
    ):

        Y_adapt = Y_new[
            adapt_indices
        ]

        return Y_old, Y_adapt

    # ---------------------------------------------------------

    @staticmethod
    def sample_weights(
        n_old,
        n_adapt
    ):

        w_old = np.ones(
            n_old
        ) / n_old

        w_adapt = np.ones(
            n_adapt
        ) / n_adapt

        return w_old, w_adapt

    # ---------------------------------------------------------

    def fit(
        self,
        Phi_old,
        Phi_new,
        Y_old,
        Y_new,
        selected_indices,
        adapt_indices
    ):

        Phi_old, Phi_adapt = self.build_design_matrix(

            Phi_old,
            Phi_new,
            selected_indices,
            adapt_indices

        )

        Y_old, Y_adapt = self.build_targets(

            Y_old,
            Y_new,
            adapt_indices

        )

        n_old = len(Y_old)
        n_adapt = len(Y_adapt)

        w_old, w_adapt = self.sample_weights(

            n_old,
            n_adapt

        )

        Phi = np.vstack(

            [

                Phi_old,
                Phi_adapt

            ]

        )

        Y = np.concatenate(

            [

                Y_old,
                Y_adapt

            ]

        )

        weights = np.concatenate(

            [

                w_old,
                w_adapt

            ]

        )

        W = np.diag(weights)

        K = Phi.shape[1]

        Sigma_inv = (

            self.alpha * np.eye(K)

            +

            self.beta *

            Phi.T @ W @ Phi

        )

        I = np.eye(K)

        Sigma = np.linalg.solve(
            Sigma_inv, I
        )

        Mu = (

            self.beta

            *

            Sigma

            @

            Phi.T

            @

            W

            @

            Y

        )

        self.posterior_mean = Mu
        self.posterior_cov = Sigma
      
        return Mu, Sigma

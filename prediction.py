"""
prediction.py
-------------

Stage 6 of the Symbolic Transfer Learning workflow.

Responsibilities
----------------
1. Bayesian prediction.
2. Predictive variance.
3. Confidence interval.
4. Performance evaluation.

"""

import numpy as np


def rmse(
    y_true,
    y_pred
):

    return np.sqrt(

        np.mean(

            (y_true-y_pred)**2

        )

    )


class BayesianPredictor:

    def __init__(
        self,
        beta=1.0
    ):

        self.beta = beta

    # ------------------------------------------------------

    def predict(

        self,

        Phi_new,

        selected_indices,

        posterior_mean,

        posterior_cov

    ):

        Phi = Phi_new[
            :,
            selected_indices
        ]

        mean = Phi @ posterior_mean

        variance = np.zeros(

            len(mean)

        )

        for i in range(

            len(mean)

        ):

            phi = Phi[i]

            variance[i] = (

                1.0/self.beta

                +

                phi

                @

                posterior_cov

                @

                phi.T

            )

        std = np.sqrt(
            variance
        )

        lower = (

            mean

            -

            1.96*std

        )

        upper = (

            mean

            +

            1.96*std

        )

        return (

            mean,
            variance,
            lower,
            upper

        )

    # ------------------------------------------------------

    def evaluate(

        self,

        y_true,

        y_pred

    ):

        score = {}

        score["RMSE"] = rmse(

            y_true,
            y_pred

        )

        score["MAE"] = np.mean(

            np.abs(

                y_true-y_pred

            )

        )

        score["R2"] = (

            1

            -

            np.sum(

                (y_true-y_pred)**2

            )

            /

            np.sum(

                (

                    y_true

                    -

                    np.mean(y_true)

                )**2

            )

        )

        return score


if __name__ == "__main__":

    np.random.seed(0)

    Phi_new = np.random.rand(

        100,
        5

    )

    posterior_mean = np.random.rand(5)

    posterior_cov = np.eye(5)*0.05

    predictor = BayesianPredictor()

    mean, var, lower, upper = predictor.predict(

        Phi_new,

        [0,1,2,3,4],

        posterior_mean,

        posterior_cov

    )

    Y = np.random.rand(100)

    metrics = predictor.evaluate(

        Y,
        mean

    )

    print(metrics)

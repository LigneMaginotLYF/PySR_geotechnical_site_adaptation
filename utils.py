# ============================================================
# utils.py
# ============================================================

import json
import logging

import numpy as np
import pandas as pd


# ------------------------------------------------------------

def load_dataset(
    filename,
    target_column=-1
):

    data = pd.read_csv(filename)

    values = data.values

    X = np.delete(
        values,
        target_column,
        axis=1
    )

    Y = values[:, target_column]

    return X.astype(float), Y.astype(float)


# ------------------------------------------------------------

def adaptation_indices(
    n_samples,
    mode="first",
    size=5,
    manual=None,
    random_seed=42
):

    if manual is None:
        manual = []

    if mode == "first":

        return np.arange(size)

    elif mode == "random":

        rng = np.random.default_rng(
            random_seed
        )

        return np.sort(

            rng.choice(
                n_samples,
                size=size,
                replace=False
            )

        )

    elif mode == "manual":

        return np.asarray(
            manual,
            dtype=int
        )

    else:

        raise ValueError(
            "Unknown adaptation mode."
        )


# ------------------------------------------------------------

def save_prediction(

    filename,

    mean,

    std,

    lower,

    upper

):

    df = pd.DataFrame({

        "Prediction": mean,

        "Std": std,

        "Lower95": lower,

        "Upper95": upper

    })

    df.to_csv(

        filename,

        index=False

    )


# ------------------------------------------------------------

def save_selected_candidates(

    filename,

    candidates,

    selected_indices

):

    rows = []

    for idx in selected_indices:

        c = candidates[idx]

        rows.append({

            "CandidateID":
                c.candidate_id,

            "Complexity":
                c.complexity,

            "RMSE_old":
                c.rmse_old,

            "RMSE_adapt":
                getattr(c, "rmse_adapt", np.nan),

            "TransferGap":
                c.transfer_gap

        })

    pd.DataFrame(rows).to_csv(

        filename,

        index=False

    )


# ------------------------------------------------------------

def save_posterior(

    output_folder,

    selected_indices,

    candidates,

    Mu,

    Sigma

):

    weights = []

    for i, idx in enumerate(selected_indices):

        weights.append({

            "CandidateID":
                candidates[idx].candidate_id,

            "Weight":
                Mu[i]

        })

    pd.DataFrame(weights).to_csv(

        output_folder /
        "posterior_mean.csv",

        index=False

    )

    np.save(

        output_folder /
        "posterior_cov.npy",

        Sigma

    )


# ------------------------------------------------------------

def save_metrics(

    filename,

    metrics

):

    with open(

        filename,

        "w"

    ) as f:

        json.dump(

            metrics,

            f,

            indent=4

        )


# ------------------------------------------------------------

def initialise_logging(

    logfile

):

    logging.basicConfig(

        filename=logfile,

        level=logging.INFO,

        format="%(asctime)s %(message)s"

    )

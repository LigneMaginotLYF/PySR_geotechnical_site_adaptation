"""
build_dictionary.py
-------------------

Stage 3 of the Symbolic Transfer Learning workflow.

Responsibilities
----------------
1. Evaluate every compiled symbolic expression.
2. Construct dictionary matrices.
3. Cache prediction vectors inside CandidateExpression.
4. Return candidate ordering.

Inputs
------
Old site:
    X_old, Y_old

Entire new site:
    X_new, Y_new

Outputs
-------
Phi_old
Phi_new
candidate_ids

Author:
"""

import numpy as np


class DictionaryBuilder:

    def __init__(self):

        pass

    # ------------------------------------------------------------

    def build(
        self,
        candidates,
        X_old,
        X_new
    ):
        """
        Parameters
        ----------
        candidates : list
            List of CandidateExpression objects.

        X_old : ndarray

        X_new : ndarray

        Returns
        -------
        Phi_old
        Phi_new
        candidate_ids
        """

        n_old = X_old.shape[0]
        n_new = X_new.shape[0]

        n_candidates = len(candidates)

        Phi_old = np.zeros(
            (n_old, n_candidates),
            dtype=float
        )

        Phi_new = np.zeros(
            (n_new, n_candidates),
            dtype=float
        )

        candidate_ids = []

        for j, candidate in enumerate(candidates):

            # -------------------------
            # Old Site
            # -------------------------

            pred_old = candidate.evaluate(X_old)

            pred_old = np.asarray(
                pred_old,
                dtype=float
            ).reshape(-1)

            # -------------------------
            # New Site
            # -------------------------

            pred_new = candidate.evaluate(X_new)

            pred_new = np.asarray(
                pred_new,
                dtype=float
            ).reshape(-1)

            # -------------------------
            # Dimension checking
            # -------------------------

            if len(pred_old) != n_old:

                raise RuntimeError(

                    f"Candidate {candidate.candidate_id} "
                    f"returned {len(pred_old)} predictions "
                    f"for old dataset with {n_old} samples."

                )

            if len(pred_new) != n_new:

                raise RuntimeError(

                    f"Candidate {candidate.candidate_id} "
                    f"returned {len(pred_new)} predictions "
                    f"for new dataset with {n_new} samples."

                )

            # -------------------------
            # Store
            # -------------------------

            Phi_old[:, j] = pred_old
            Phi_new[:, j] = pred_new

            candidate.prediction_old = pred_old
            candidate.prediction_new = pred_new

            candidate_ids.append(
                candidate.candidate_id
            )

        return (
            Phi_old,
            Phi_new,
            candidate_ids
        )

    # ------------------------------------------------------------

    @staticmethod
    def subset(
        Phi,
        indices
    ):
        """
        Obtain a subset of rows.

        Intended for extracting the
        adaptation dataset from the
        complete new-site dictionary.

        Parameters
        ----------
        Phi : ndarray

        indices : array-like

        Returns
        -------
        ndarray
        """

        return Phi[indices, :]

    # ------------------------------------------------------------

    @staticmethod
    def select_candidates(
        Phi,
        columns
    ):
        """
        Select surviving candidate columns.

        Parameters
        ----------
        Phi : ndarray

        columns : array-like

        Returns
        -------
        ndarray
        """

        return Phi[:, columns]


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    class DummyCandidate:

        def __init__(self, idx):

            self.candidate_id = idx

        def evaluate(self, X):

            return np.sum(
                X,
                axis=1
            ) * (self.candidate_id + 1)


    np.random.seed(0)

    X_old = np.random.rand(
        100,
        6
    )

    X_new = np.random.rand(
        50,
        6
    )

    candidates = [

        DummyCandidate(i)

        for i in range(8)

    ]

    builder = DictionaryBuilder()

    Phi_old, Phi_new, ids = builder.build(

        candidates,
        X_old,
        X_new

    )

    print(Phi_old.shape)
    print(Phi_new.shape)
    print(ids)

    # Example adaptation subset

    subset_indices = [0, 1, 2, 5, 8]

    Phi_adapt = builder.subset(
        Phi_new,
        subset_indices
    )

    print(Phi_adapt.shape)

    # Example selected candidates

    selected = [0, 3, 4]

    Phi_selected = builder.select_candidates(
        Phi_new,
        selected
    )

    print(Phi_selected.shape)

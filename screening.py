"""
screening.py
------------

Stage 4 of the Symbolic Transfer Learning workflow.

Workflow
--------
Filter 1
    Pareto front extraction

↓

Filter 2
    Hierarchical clustering on Phi_new

↓

Representative selection
    Lowest complexity within each cluster

↓

Filter 3
    Rank by transferability

Outputs
-------
Selected candidate indices
Selected CandidateExpression objects

Author:
"""

import numpy as np

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform


# ============================================================
# Metrics
# ============================================================

def rmse(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


# ============================================================
# Screening
# ============================================================

class CandidateScreening:

    def __init__(self):

        pass

    # --------------------------------------------------------

    def evaluate_candidates(
        self,
        candidates,
        Y_old,
        Y_new
    ):

        """
        Compute candidate metrics.
        """

        for candidate in candidates:

            candidate.rmse_old = rmse(

                Y_old,
                candidate.prediction_old

            )

            candidate.rmse_new = rmse(

                Y_new,
                candidate.prediction_new

            )

            candidate.transfer_gap = abs(

                candidate.rmse_old
                -
                candidate.rmse_new

            )

    # --------------------------------------------------------

    def pareto_fronts(
        self,
        candidates
    ):

        """
        Non-dominated sorting.

        Objectives

            RMSE_old
            Complexity

        Returns
        -------
        list of fronts
        """

        remaining = list(range(len(candidates)))

        fronts = []

        while len(remaining) > 0:

            front = []

            for i in remaining:

                dominated = False

                for j in remaining:

                    if i == j:
                        continue

                    ci = candidates[i]
                    cj = candidates[j]

                    better_or_equal = (

                        cj.rmse_old <= ci.rmse_old

                        and

                        cj.complexity <= ci.complexity

                    )

                    strictly_better = (

                        cj.rmse_old < ci.rmse_old

                        or

                        cj.complexity < ci.complexity

                    )

                    if better_or_equal and strictly_better:

                        dominated = True
                        break

                if not dominated:

                    front.append(i)

            fronts.append(front)

            remaining = [

                i

                for i in remaining

                if i not in front

            ]

        return fronts

    # --------------------------------------------------------

    def keep_fronts(

        self,

        fronts,

        n_fronts=3

    ):

        keep = []

        for f in fronts[:n_fronts]:

            keep.extend(f)

        return keep

    # --------------------------------------------------------

    def cluster_candidates(

        self,

        Phi_new,

        candidate_indices

    ):

        """
        Hierarchical clustering.

        Returns

            cluster labels
        """

        Phi = Phi_new[:, candidate_indices]

        correlation = np.corrcoef(
            Phi.T
        )

        correlation = np.nan_to_num(
            correlation
        )

        distance = 1.0 - np.abs(
            correlation
        )

        np.fill_diagonal(
            distance,
            0.0
        )

        condensed = squareform(
            distance,
            checks=False
        )

        Z = linkage(

            condensed,

            method="average"

        )

        return Z

    # --------------------------------------------------------

    def representatives(

        self,

        linkage_matrix,

        candidates,

        kept_indices,

        target_candidates=5

    ):

        """
        Determine clusters automatically.

        The threshold is increased until
        the number of clusters becomes
        approximately target_candidates.
        """

        thresholds = np.linspace(

            0.01,
            1.0,
            200

        )

        labels = None

        for t in thresholds:

            trial = fcluster(

                linkage_matrix,

                t=t,

                criterion="distance"

            )

            n_clusters = len(
                np.unique(trial)
            )

            labels = trial

            if n_clusters <= target_candidates:

                break

        selected = []

        for cluster in np.unique(labels):

            members = np.where(
                labels == cluster
            )[0]

            best = None

            for m in members:

                idx = kept_indices[m]

                c = candidates[idx]

                if best is None:

                    best = idx

                    continue

                b = candidates[best]

                if c.complexity < b.complexity:

                    best = idx

            selected.append(best)

        return selected

    # --------------------------------------------------------

    def transferability(

        self,

        candidates,

        selected,

        target_candidates=5

    ):

        """
        Final ranking.
        """

        selected = sorted(

            selected,

            key=lambda i: (

                candidates[i].transfer_gap,

                candidates[i].complexity

            )

        )

        return selected[:target_candidates]

    # --------------------------------------------------------

    def run(

        self,

        candidates,

        Phi_new,

        Y_old,

        Y_new,

        pareto_fronts=3,

        target_candidates=5

    ):

        """
        Complete screening.
        """

        self.evaluate_candidates(

            candidates,

            Y_old,

            Y_new

        )

        fronts = self.pareto_fronts(
            candidates
        )

        kept = self.keep_fronts(

            fronts,

            pareto_fronts

        )

        linkage_matrix = self.cluster_candidates(

            Phi_new,

            kept

        )

        representatives = self.representatives(

            linkage_matrix,

            candidates,

            kept,

            target_candidates

        )

        selected = self.transferability(

            candidates,

            representatives,

            target_candidates

        )

        selected_candidates = [

            candidates[i]

            for i in selected

        ]

        return (

            selected,

            selected_candidates

        )


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    class DummyCandidate:

        def __init__(

            self,

            idx,

            complexity

        ):

            self.candidate_id = idx

            self.complexity = complexity

            self.prediction_old = np.random.rand(100)

            self.prediction_new = np.random.rand(50)


    np.random.seed(1)

    candidates = [

        DummyCandidate(

            i,

            np.random.randint(3,30)

        )

        for i in range(40)

    ]

    Phi_new = np.random.rand(

        50,

        40

    )

    Y_old = np.random.rand(100)

    Y_new = np.random.rand(50)

    screening = CandidateScreening()

    ids, selected = screening.run(

        candidates,

        Phi_new,

        Y_old,

        Y_new,

        pareto_fronts=3,

        target_candidates=5

    )

    print(ids)

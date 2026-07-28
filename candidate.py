"""
candidate.py

Core data structure for symbolic expressions extracted from PySR.

"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class CandidateExpression:

    # ---------- Basic information ----------
    candidate_id: int
    equation: str
    complexity: float
    loss: float

    # ---------- Source ----------
    source_folder: str
    source_file: str
    run_name: str

    # ---------- Compiled function ----------
    function: Optional[Callable] = None

    # ---------- Cached prediction ----------
    cached_prediction: Optional[np.ndarray] = None

    # ---------- Screening ----------
    selected: bool = False
    selection_frequency: float = 0.0

    # ---------- Bayesian ----------
    posterior_weight: float = 0.0
    posterior_probability: float = 0.0

    # ---------- Optional notes ----------
    notes: str = ""

    def evaluate(self, X):
        """
        Evaluate the symbolic expression.

        Parameters
        ----------
        X : ndarray
            Feature matrix.

        Returns
        -------
        ndarray
        """
        if self.function is None:
            raise RuntimeError(
                f"Candidate {self.candidate_id} has not been compiled."
            )

        return self.function(X)

    def __repr__(self):

        return (
            f"Candidate("
            f"id={self.candidate_id}, "
            f"loss={self.loss:.4e}, "
            f"complexity={self.complexity}, "
            f"selected={self.selected})"
        )

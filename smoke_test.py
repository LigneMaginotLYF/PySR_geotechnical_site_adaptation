import numpy as np
from pathlib import Path
import pandas as pd

import config
from extract_pysr import PySRCandidateExtractor
from parser import ExpressionParser
from build_dictionary import DictionaryBuilder
from screening import CandidateScreening
from bayesian_regression import BayesianLinearRegression
from prediction import BayesianPredictor
from utils import load_dataset, adaptation_indices

def main():
    # quick precheck
    assert Path(config.OLD_DATASET).exists(), f"Missing {config.OLD_DATASET}"
    assert Path(config.NEW_DATASET).exists(), f"Missing {config.NEW_DATASET}"
    assert Path(config.PYSR_FOLDER).exists(), f"Missing {config.PYSR_FOLDER}"

    X_old, Y_old = load_dataset(config.OLD_DATASET, config.TARGET_COLUMN)
    X_new, Y_new = load_dataset(config.NEW_DATASET, config.TARGET_COLUMN)

    adapt_idx = adaptation_indices(
        len(X_new),
        mode=config.ADAPT_MODE,
        size=config.ADAPT_SIZE,
        manual=config.MANUAL_INDICES,
        random_seed=config.RANDOM_SEED,
    )

    extractor = PySRCandidateExtractor()
    candidates = extractor.search(config.PYSR_FOLDER)
    assert len(candidates) > 0, "No candidates extracted"

    parser = ExpressionParser()
    for c in candidates:
        parser.compile_candidate(c)

    builder = DictionaryBuilder()
    Phi_old, Phi_new, _ = builder.build(candidates, X_old, X_new)

    screening = CandidateScreening()
    selected_indices, _ = screening.run(
        candidates, Phi_new, Y_old, Y_new, adapt_idx=adapt_idx,
        pareto_fronts=config.N_PARETO_FRONTS,
        target_candidates=config.TARGET_CANDIDATES,
    )
    assert len(selected_indices) > 0, "No candidates selected"

    bayes = BayesianLinearRegression(alpha=config.ALPHA, beta=config.BETA)
    Mu, Sigma = bayes.fit(Phi_old, Phi_new, Y_old, Y_new, selected_indices, adapt_idx)

    predictor = BayesianPredictor(beta=config.BETA)
    mean, std, lower, upper = predictor.predict(Phi_new, selected_indices, Mu, Sigma)
    assert mean.shape[0] == X_new.shape[0]
    assert np.all(np.isfinite(mean)), "Non-finite predictions"

    metrics = predictor.evaluate(Y_new, mean)
    print("Smoke test passed.")
    print(metrics)

if __name__ == "__main__":
    main()

# ============================================================
# main.py
# ============================================================

import logging

import config

from utils import (
    load_dataset,
    adaptation_indices,
    save_prediction,
    save_selected_candidates,
    save_posterior,
    save_metrics,
    initialise_logging
)

from stage1_extract import CandidateExtractor
from stage2_parser import ExpressionParser
from stage3_dictionary import DictionaryBuilder
from stage4_screening import CandidateScreening
from stage5_bayesian import BayesianLinearRegression
from stage6_prediction import BayesianPredictor


# ============================================================

def main():

    initialise_logging(

        config.OUTPUT_FOLDER /
        "experiment.log"

    )

    logging.info("Loading datasets...")

    X_old, Y_old = load_dataset(

        config.OLD_DATASET,

        config.TARGET_COLUMN

    )

    X_new, Y_new = load_dataset(

        config.NEW_DATASET,

        config.TARGET_COLUMN

    )

    adapt_idx = adaptation_indices(

        len(X_new),

        mode=config.ADAPT_MODE,

        size=config.ADAPT_SIZE,

        manual=config.MANUAL_INDICES,

        random_seed=config.RANDOM_SEED

    )

    logging.info("Stage 1")

    extractor = CandidateExtractor()

    candidates = extractor.extract(

        config.PYSR_FOLDER

    )

    logging.info(

        f"{len(candidates)} candidates extracted."

    )

    logging.info("Stage 2")

    parser = ExpressionParser()

    for c in candidates:

        parser.compile_candidate(c)

    logging.info("Stage 3")

    builder = DictionaryBuilder()

    Phi_old, Phi_new, candidate_ids = builder.build(

        candidates,

        X_old,

        X_new

    )

    logging.info("Stage 4")

    screening = CandidateScreening()

    selected_indices, selected_candidates = screening.run(

        candidates,

        Phi_new,

        Y_old,

        Y_new,

        pareto_fronts=config.N_PARETO_FRONTS,

        target_candidates=config.TARGET_CANDIDATES

    )

    logging.info(

        f"Selected {len(selected_indices)} candidates."

    )

    logging.info("Stage 5")

    bayes = BayesianLinearRegression(

        alpha=config.ALPHA,

        beta=config.BETA

    )

    Mu, Sigma = bayes.fit(

        Phi_old,

        Phi_new,

        Y_old,

        Y_new,

        selected_indices,

        adapt_idx

    )

    logging.info("Stage 6")

    predictor = BayesianPredictor(

        beta=config.BETA

    )

    mean, std, lower, upper = predictor.predict(

        Phi_new,

        selected_indices,

        Mu,

        Sigma

    )

    metrics = predictor.evaluate(

        Y_new,

        mean

    )

    logging.info(metrics)

    save_prediction(

        config.OUTPUT_FOLDER /
        "prediction.csv",

        mean,

        std,

        lower,

        upper

    )

    save_selected_candidates(

        config.OUTPUT_FOLDER /
        "selected_candidates.csv",

        candidates,

        selected_indices

    )

    save_posterior(

        config.OUTPUT_FOLDER,

        selected_indices,

        candidates,

        Mu,

        Sigma

    )

    save_metrics(

        config.OUTPUT_FOLDER /
        "metrics.json",

        metrics

    )

    logging.info("Finished.")


# ============================================================

if __name__ == "__main__":

    main()

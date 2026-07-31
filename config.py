# ============================================================
# config.py
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Project
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

PYSR_FOLDER = PROJECT_ROOT / "datasets" / "tunnel heave case" / "pysr_results"

OLD_DATASET = PROJECT_ROOT / "datasets" / "tunnel heave case" / "data" / "old_dimless_5.csv"

NEW_DATASET = PROJECT_ROOT / "datasets" / "tunnel heave case" / "data" / "new_dimless_5.csv"

TARGET_COLUMN = -1

# ------------------------------------------------------------
# Adaptation
# ------------------------------------------------------------

ADAPT_MODE = "random"
# first | random | manual

ADAPT_SIZE = 20

MANUAL_INDICES = [0, 14, 33, 49]  # 1-14, 1-19, 1-16, 1-14 -> 1,15,34,50

RANDOM_SEED = 42

# ------------------------------------------------------------
# Screening
# ------------------------------------------------------------

N_PARETO_FRONTS = 2

TARGET_CANDIDATES = 20

# ------------------------------------------------------------
# Bayesian
# ------------------------------------------------------------

ALPHA = 1.0

BETA = 100

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_FOLDER = PROJECT_ROOT / "output"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

"""
extract_pysr.py

Read every PySR hall_of_fame.csv.

"""

import os
import pandas as pd

from candidate import CandidateExpression


class PySRCandidateExtractor:

    def __init__(self):

        self.candidates = []

    def search(self, root_folder):

        counter = 0

        for root, dirs, files in os.walk(root_folder):

            for file in files:

                if file.endswith(".csv") and "hall" in file.lower():

                    csv_path = os.path.join(root, file)

                    print(f"Loading {csv_path}")

                    df = pd.read_csv(csv_path)

                    cols = [c.lower() for c in df.columns]

                    # Locate columns automatically
                    eq_col = df.columns[
                        cols.index(
                            next(c for c in cols if "equation" in c)
                        )
                    ]

                    loss_col = df.columns[
                        cols.index(
                            next(c for c in cols if "loss" in c)
                        )
                    ]

                    complexity_col = df.columns[
                        cols.index(
                            next(c for c in cols if "complex" in c)
                        )
                    ]

                    for _, row in df.iterrows():

                        candidate = CandidateExpression(

                            candidate_id=counter,

                            equation=str(row[eq_col]),

                            complexity=float(row[complexity_col]),

                            loss=float(row[loss_col]),

                            source_folder=root,

                            source_file=file,

                            run_name=os.path.basename(root)

                        )

                        self.candidates.append(candidate)

                        counter += 1

        print(f"Total candidates : {len(self.candidates)}")

        return self.candidates

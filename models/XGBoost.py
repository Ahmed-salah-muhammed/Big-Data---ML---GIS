"""
DEPRECATED — kept only to surface a helpful message if someone runs it.

This script previously trained a single `MultiOutputClassifier(XGBClassifier)`
on both targets and saved it as `xgboosst.pkl` (note the typo). Several
problems made it the wrong tool for this project:

1. It read the dataset from `data/final_dataset.csv` — the correct path is
   `data/processed/final_dataset.csv`. So it would crash on a clean checkout.
2. `MultiOutputClassifier` triggers, on modern scikit-learn, the warning:
       UserWarning: `sklearn.utils.parallel.delayed` should be used with
       `sklearn.utils.parallel.Parallel` ...
   That is the warning you reported.
3. It saved to `traffic_model.pkl`, colliding with `train_model.py`'s output
   (which is a single-output classifier with a different feature set). Whoever
   ran the two scripts in the wrong order would silently load the wrong file.
4. It used `train_test_split` without `random_state` or `stratify`, so reruns
   produced different test sets and unstable metrics.

Use `python models/train_model.py` instead. It trains both models cleanly,
saves them to distinct filenames, and emits no warnings.
"""

import sys


if __name__ == "__main__":
    print(__doc__)
    sys.exit(
        "\nThis script has been deprecated. Run instead:\n"
        "    python models/train_model.py\n"
    )

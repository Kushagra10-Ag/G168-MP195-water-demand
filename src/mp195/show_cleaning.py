
"""Show what the cleaning did to train.csv (owner: Kushagra)."""

import sys
import pandas as pd

sys.path.insert(0, "src")

from mp195.cleaning import clean_train

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

clean, log = clean_train(train, later, holdout)

for key, value in log.items():
    print(f"{key}: {value}")

print("\nMissing values left after cleaning (the model fills these):")

print(clean.isna().sum()[clean.isna().sum() > 0])

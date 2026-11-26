import sys
from pprint import pprint
import pandas as pd

sys.path.insert(0, "src")

from mp195 import data_checks_a, data_checks_b, data_checks_c, data_checks_d

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

summary = {}

summary.update(data_checks_a.report(train))
summary.update(data_checks_b.report(train, holdout))
summary.update(data_checks_c.report(train, later))
summary.update(data_checks_d.report(train))

pprint(summary, sort_dicts=False)
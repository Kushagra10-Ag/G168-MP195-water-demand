
import pandas as pd

from mp195.data_checks_c import (
    find_spelling_variants,
    find_unit_mixup
)


def test_unit_mixup_finds_the_big_tower():
    df = pd.DataFrame({
        "tower": ["A", "A", "B", "B", "C", "C"],
        "garden": [100, 110, 1076.4, 1184.04, 105, 95],
    })

    result = find_unit_mixup(df)

    assert result[0]["group_value"] == "B"
    assert result[0]["factor"] == 10.764


def test_spelling_variants_found():
    train = pd.DataFrame({
        "src": ["rain", "Rain ", "dry", "dry"]
    })

    clean = pd.DataFrame({
        "src": ["rain", "dry"]
    })

    result = find_spelling_variants(train, clean)

    assert result[0]["true_levels"] == 2
    assert "Rain " in result[0]["bad_spellings"]

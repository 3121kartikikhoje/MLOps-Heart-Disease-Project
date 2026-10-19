import pandas as pd
from src.data import COLUMNS

def test_dataset_column_definition():
    assert len(COLUMNS) == 14
    assert COLUMNS[-1] == "target"

def test_binary_target_conversion_logic():
    target = pd.Series([0, 1, 2, 3, 4])
    binary = (target > 0).astype(int)
    assert binary.tolist() == [0, 1, 1, 1, 1]

import pandas as pd
import sys, os
from src.analysis_utils import compute_yearly_averages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_compute_yearly_averages():
    df = pd.DataFrame({
        "Jan": [10, 20],
        "Feb": [20, 40],
        "Mar": [30, 60]
    }, index=[2000, 2001])

    expected = [20.0, 40.0]
    result = compute_yearly_averages(df)

    assert result == expected

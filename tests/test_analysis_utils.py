import pandas as pd
import numpy as np
from src.analysis_utils import compute_yearly_averages, fit_polynomial_models

def test_compute_yearly_averages():
    df = pd.DataFrame({
        "Jan": [10, 20],
        "Feb": [20, 40],
        "Mar": [30, 60]
    }, index=[2000, 2001])

    expected = [20.0, 40.0]
    result = compute_yearly_averages(df)

    assert result == expected

def test_polynomial_model_output_structure():
    X_train = np.array([0, 1, 2, 3])
    y_train = np.array([1, 2, 3, 4])
    X_test = np.array([4, 5])
    y_test = np.array([5, 6])

    polynomials, mse_values, _, _, _ = fit_polynomial_models(X_train, y_train, X_test, y_test)

    # Correct number of models
    assert len(polynomials) == 3
    assert len(mse_values) == 3

    # Keys should match orders
    assert set(polynomials.keys()) == {1, 2, 3}
    assert set(mse_values.keys()) == {1, 2, 3}

def test_polynomial_models_are_callable():
    X_train = np.array([0, 1, 2])
    y_train = np.array([0, 1, 4])
    X_test = np.array([3])
    y_test = np.array([9])

    polynomials, _, _, _, _ = fit_polynomial_models(X_train, y_train, X_test, y_test)

    for model in polynomials.values():
        assert callable(model)

def test_mse_values_are_floats():
    X_train = np.array([0, 1, 2])
    y_train = np.array([0, 1, 4])
    X_test = np.array([3])
    y_test = np.array([9])

    _, mse_values, _, _, _ = fit_polynomial_models(X_train, y_train, X_test, y_test)

    for mse in mse_values.values():
        assert isinstance(mse, float)

def test_perfect_polynomial_fit():
    # Perfect quadratic: y = x^2
    X_train = np.array([0, 1, 2, 3])
    y_train = X_train**2

    X_test = np.array([4, 5])
    y_test = X_test**2

    _, mse_values, _, _, _ = fit_polynomial_models(X_train, y_train, X_test, y_test, orders=[1, 2, 3])

    # Degree 2 should be perfect
    assert mse_values[2] < 1e-10

    # Degree 1 should be worse
    assert mse_values[1] > mse_values[2]

def test_inputs_not_modified():
    X_train = np.array([0, 1, 2])
    y_train = np.array([0, 1, 4])
    X_test = np.array([3])
    y_test = np.array([9])

    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()

    fit_polynomial_models(X_train, y_train, X_test, y_test)

    assert np.array_equal(X_train, X_train_copy)
    assert np.array_equal(y_train, y_train_copy)


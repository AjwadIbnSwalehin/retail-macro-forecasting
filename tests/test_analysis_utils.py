import pandas as pd
import numpy as np
from src.analysis_utils import compute_yearly_averages, fit_polynomial_models, calculate_forecast, calculate_training_data

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


def test_calculate_training_data_basic_split():
    x = np.array([2000, 2001, 2002, 2003, 2004])
    y = np.array([10, 20, 30, 40, 50])

    train_size, X_train, y_train, X_test, y_test = calculate_training_data(x, y, 2)

    assert train_size == 3
    assert np.array_equal(X_train, np.array([0, 1, 2]))  # scaled years
    assert np.array_equal(y_train, np.array([10, 20, 30]))
    assert np.array_equal(X_test, np.array([3, 4]))
    assert np.array_equal(y_test, np.array([40, 50]))


def test_calculate_training_data_scaling():
    x = np.array([1995, 2000, 2010])
    y = np.array([5, 10, 15])

    _, X_train, _, _, _ = calculate_training_data(x, y, 1)

    # scaled: subtract min (1995)
    assert np.array_equal(X_train, np.array([0, 5]))


def test_calculate_training_data_number_of_tests_zero():
    x = np.array([1, 2, 3])
    y = np.array([10, 20, 30])

    train_size, X_train, y_train, X_test, y_test = calculate_training_data(x, y, 0)

    assert train_size == 3
    assert len(X_test) == 0
    assert len(y_test) == 0


def test_calculate_forecast_basic():
    # simple linear model: y = x
    model = lambda x: x

    y_test = np.array([1, 2, 3])
    predictions = {1: np.array([1, 2, 3])}  # perfect fit
    future_years = np.array([4, 5])
    years = np.array([1, 2, 3])
    best_order = 1

    future_preds, lower, upper = calculate_forecast(
        y_test, best_order, future_years, years, model, predictions
    )

    # perfect fit → residuals = 0 → std_error = 0
    assert np.array_equal(future_preds, np.array([3, 4]))  # scaled: future_years - min(years)
    assert np.array_equal(lower, future_preds)
    assert np.array_equal(upper, future_preds)


def test_calculate_forecast_with_residuals():
    model = lambda x: x

    y_test = np.array([2, 4, 6])
    predictions = {1: np.array([1, 3, 5])}  # residuals = [1,1,1]
    future_years = np.array([4])
    years = np.array([1, 2, 3])
    best_order = 1

    future_preds, lower, upper = calculate_forecast(
        y_test, best_order, future_years, years, model, predictions
    )

    # std_error = 0 because residuals are constant
    assert future_preds == 3
    assert lower == 3
    assert upper == 3

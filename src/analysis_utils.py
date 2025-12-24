from sklearn.metrics import mean_squared_error
import numpy as np


def compute_yearly_averages(df):
    yearly_averages = []
    for year in df.index:
        average = df.loc[year].mean()
        yearly_averages.append(average)
    return yearly_averages


def fit_polynomial_models(X_train, y_train, X_test, y_test, orders=[1, 2, 3]):
    polynomials = {}
    mse_values = {}

    # Fit models
    for order in orders:
        coeffs = np.polyfit(X_train, y_train, order)
        model = np.poly1d(coeffs)

        y_pred = model(X_test)
        mse = mean_squared_error(y_test, y_pred)

        polynomials[order] = model
        mse_values[order] = mse
    
    return polynomials, mse_values
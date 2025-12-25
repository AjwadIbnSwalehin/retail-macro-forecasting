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
    chi2_values = {}
    bic_values = {}
    predictions = {}

    for order in orders:
        coeffs = np.polyfit(X_train, y_train, order)
        model = np.poly1d(coeffs)

        y_pred = model(X_test)
        predictions[order] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        mse_values[order] = mse

        residuals = y_test - y_pred
        chi2 = np.sum(residuals**2)
        dof = len(y_test) - (order + 1)
        chi2_values[order] = chi2 / dof

        bic = len(y_test) * np.log(np.mean(residuals**2)) + (order + 1) * np.log(len(y_test))
        bic_values[order] = bic

        polynomials[order] = model

    return polynomials, mse_values, predictions, chi2_values, bic_values

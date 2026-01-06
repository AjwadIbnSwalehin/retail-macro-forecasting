# retail-macro-forecasting
This project investigates long-term trends in U.S. retail and food services sales and explores how macroeconomic, specifically inflation (CPI) and the unemployment rate relate to consumer spending.

## Data Sources
All datasets in `data/` are publicly available

- Unemployment rate: `civilian_unemployment_rate.csv`
  - Source: U.S. Census Bureau
- Inflation Rate: `inflation_rate.csv`
  - U.S. Bureau of Labor Statistics
- Retail Sales: `retail_and_food_serices_adjusted,csv`
  - U.S. Bureau of Labor Statistics

## Data Preprocessing
Key preprocessing steps include:
- Converting monthly retail data into yearly averages
- Removing percentage symbols from CPI values
- Selecting only the “Total” unemployment column
- Scaling year values to improve numerical stability in polynomial fitting
- Splitting each dataset into training and test sets (e.g., 18/3 or 28/5)

## Modelling Approach
Polynomial models of degree 1, 2 and 3 were fitted to each dataset using NumPy's `polyfit`.
Models were evaluated using:
- Mean Squared Error (MSE)
- Chi-Squared per Degree of Freedom (χ²/DoF)
- Bayesian Information Criterion (BIC)

## Forecasting
The selected models were used to forecast values for **2026-2030**, with **95% confidence intervals** constructed using residual standard deviation from the test set.
Forecasts include:
- Retail & Food Services Sales
- Unemployment Rate
- CPI Rate

## Unit Tests
Unit tests found in `tests/` are included functions found in `src/analysis_utils.py`:
- `compute_yearly_averages`
- `fit_polynomial_models`
- `calculate_training_data`
- `calculate_forecast`
These tests validate correct splitting, scaling, forecasting, and confidence interval behaviour.
Run tests with:

`pytest`

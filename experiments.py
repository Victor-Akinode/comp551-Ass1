import numpy as np
import pandas as pd
from linear_regression import LinearRegression
from data_parser import process_csv, ArrayScaler, OutlierHandler
from feature_engineering import add_nonlinear_features

def mse(y, yhat):

    return float(np.mean((y - yhat) ** 2))

def run_experiments(csv_path):
    
    rows = []

    lambdas = [0.0, 0.1, 1.0]

    option = [False, True]
    outlier_methods = ["none", "quantile", "iqr"]

    for polynomial in option:
        for interaction in option:
            for transform in option:
                
                if transform:
                    one_hot_cols = ["season", "weekday"]
                else:
                    one_hot_cols = ["season", "mnth", "weekday"]

                X_train, X_test, y_train, y_test, fi = process_csv(csv_path, one_hot_encode = one_hot_cols, scaling_method = "none", outlier_method = "none")

                X_train, fi = add_nonlinear_features(
                    X_train, fi,
                    polynomial = polynomial,
                    interaction = interaction,
                    transform = transform)
                
                X_test, _ = add_nonlinear_features(
                    X_test, fi,
                    polynomial = polynomial,
                    interaction = interaction,
                    transform = transform)

                engineered_names = []
                if polynomial:
                    engineered_names += ["temp_sq", "hum_sq", "windspeed_sq"]
                if interaction:
                    engineered_names += ["temp_hum", "temp_windspeed", "temp_weathersit"]
                if transform:
                    engineered_names += ["month_sin", "month_cos"]

                engineered_idx = [fi[name] for name in engineered_names if name in fi]
                continuous_idx = [fi[name] for name in ["temp", "hum", "windspeed"]] + engineered_idx

                for outlier_method in outlier_methods:
                    for lam in lambdas:
                        Xtr = X_train.copy()
                        Xte = X_test.copy()

                        if True:
                            handler = OutlierHandler(method = outlier_method, idx = continuous_idx)
                            handler.fit(Xtr)
                            Xtr = handler.transform(Xtr)
                            Xte = handler.transform(Xte)

                        if True:
                            scaler = ArrayScaler(method = "zscore", idx = continuous_idx)
                            scaler.fit(Xtr)
                            Xtr = scaler.transform(Xtr)
                            Xte = scaler.transform(Xte)

                        model = LinearRegression()
                        model.fit(Xtr, y_train, lam=lam)

                        ytr_hat = model.predict(Xtr)
                        yte_hat = model.predict(Xte)

                        rows.append({
                            "poly": polynomial,
                            "inter": interaction,
                            "month_sincos": transform,
                            "clip": outlier_method,
                            "lambda": lam,
                            "train_root_mse": np.sqrt(mse(y_train, ytr_hat)),
                            "test_root_mse": np.sqrt(mse(y_test, yte_hat)),
                            "gap": np.sqrt(mse(y_test, yte_hat)) - np.sqrt(mse(y_train, ytr_hat)),
                            "n_features": Xtr.shape[1]})

    results = pd.DataFrame(rows).sort_values("test_root_mse", ascending=True)
    return results

if __name__ == "__main__":

    print(run_experiments("data/day.csv").to_string())
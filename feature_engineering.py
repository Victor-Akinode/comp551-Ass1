import numpy as np

def add_nonlinear_features(X, temp_idx, hum_idx, wind_idx):
    """
    Adds polynomial and interaction features for selected continuous columns.

    Parameters:
    - X: NumPy array (already includes bias column)
    - temp_idx, hum_idx, wind_idx: indices of continuous features

    Returns:
    - X_new: expanded feature matrix
    """

    # Extract continuous features
    temp = X[:, temp_idx]
    hum = X[:, hum_idx]
    wind = X[:, wind_idx]

    # Polynomial features
    temp_sq = temp ** 2
    hum_sq = hum ** 2
    wind_sq = wind ** 2

    # Interaction features
    temp_hum = temp * hum
    temp_wind = temp * wind
    hum_wind = hum * wind

    # Concatenate original features with new ones
    X_new = np.column_stack([
        X,
        temp_sq,
        hum_sq,
        wind_sq,
        temp_hum,
        temp_wind,
        hum_wind
    ])

    return X_new

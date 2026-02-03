import numpy as np

def add_nonlinear_features(X, feature_index, polynomial = True, interaction = True, transform = True):
    """
    Adds polynomial and interaction features for selected continuous columns.

    Parameters:
    - X: NumPy array (already includes bias column)
    - temp_idx, hum_idx, wind_idx: indices of continuous features

    Returns:
    - X_new: expanded feature matrix
    """

    fi = dict(feature_index)
    new_cols = []
    new_names = []

    if transform and "mnth" not in fi:

        raise ValueError("Please ensure that mnth was not one-hot encoded before attempting to perform sin/cos transform.")    

    # Extract features
    temp = X[:, fi["temp"]]
    hum = X[:, fi["hum"]]
    windspeed = X[:, fi["windspeed"]]
    weathersit = X[:, fi["weathersit"]]
    if transform:
        mnth = X[:, fi["mnth"]]

    # Polynomial features
    if polynomial:

        new_cols += [temp**2, hum**2, windspeed**2]
        new_names += ["temp_sq", "hum_sq", "windspeed_sq"]

    # Interaction features
    if interaction:

        new_cols += [temp * hum, temp * windspeed, temp * weathersit]
        new_names += ["temp_hum", "temp_windspeed", "temp_weathersit"]
    
    # Transformation of month feature
    if transform:

        theta = 2 * np.pi * (mnth - 1) / 12.0
        new_cols += [np.sin(theta), np.cos(theta)]
        new_names += ["month_sin", "month_cos"]

    if not new_cols:
    
        return X, fi
    
    # Concatenate original features with new ones
    X_new = np.column_stack([X] + new_cols)

    # Update feature index
    start_idx = X.shape[1]
    for i, name in enumerate(new_names):

        fi[name] = start_idx + i

    return X_new, fi
import pandas as pd
import numpy as np

def process_csv(csv_path, train_frac = 0.8, scaling_method = "zscore", outlier_method = "quantile", one_hot_encode = ["mnth", "season", "weekday"]):
    """
    Takes path to day.csv from the UCI bike sharing dataset and returns (X_train, X_test, y_train, y_test, feature_index) as NumPy
    arrays except feature_index which is a dictionary from feature name to index.

    This implementations drops the following features: ["instant", "casual", "registered", "workingday", "atemp", "dteday"] due
    to relevancy, data leakage, and potential for colinearity. A full justification is provided in the complementary writeup.pdf.

    Selected non-binary categorical features will be one-hot encoded and all real-valued features are standardized based on the training
    dataset to have mean 0 and variance 1 (this same transformation is applied to both X_train and X_test).

    Due to the time-series like nature of the data, the train/test split is time-aware with approximately the first train_frac
    fraction of entries alloted to the training set and approximately the last 1 - train_frac fraction of entries alloted to the
    testing set. The entries were manually and explicitly sorted by date using the "dteday" column before dropping it in order to
    achieve this for the sake of sureness, despite the dataset seeming to already be sorted by date in the correct order.
    
    Please note that the column names in preprocessing operations are hard-coded and assumed to match the specifications of the
    dataset. If this is not the case, a value error will be raised. (TO-DO: make error highlight which entries are problematic?)
    """

    df = pd.read_csv(csv_path)

    # Drop all undesired columns as described in writeup.pdf, except "dteday" which will be used for sorting before later being dropped
    df = df.drop(columns=["instant", "casual", "registered", "workingday", "atemp"])

    # Validate that all data is conforming to the specifications of the dataset and that no entries are missing
    if (
        (~df["holiday"].isin([0, 1])).any()
        or (~df["temp"].between(0, 1)).any()
        or (~df["windspeed"].between(0, 1)).any()
        or (~df["hum"].between(0, 1)).any()
        or (~df["season"].isin(list(range(1,5)))).any()
        or (~df["yr"].isin([0, 1])).any()
        or (~df["mnth"].isin(list(range(1,13)))).any()
        or (~df["weekday"].isin(list(range(0,7)))).any()
        or (~df["weathersit"].isin(list(range(1, 5)))).any()
        or (~df["cnt"] >= 0).any()
        or df.isnull().sum().sum() > 0
    ):
        raise ValueError("Missing or non-conforming values were found; please ensure all fields are populated as expected according to the specifications of the dataset.")

    # One-hot encode selected categorical features (first converting them to pd.Categorical so that the column names are clearer for debugging purposes)
    for feature in one_hot_encode:
        if feature not in ["mnth", "season", "weekday", "weathersit"]:
            raise ValueError("Can only one-hot encode categorical features which are mnth, season, weekday, and weathersit.")
    df["mnth"] = pd.Categorical(df["mnth"], categories = range(1, 13)).rename_categories(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    df["season"] = pd.Categorical(df["season"], categories = [1, 2, 3, 4]).rename_categories(["winter", "spring", "summer", "fall"])
    df["weekday"] = pd.Categorical(df["weekday"], categories = range(7)).rename_categories(["sun", "mon", "tue", "wed", "thu", "fri", "sat"])
    df["weathersit"] = pd.Categorical(df["weathersit"], categories = [1, 2, 3, 4]).rename_categories(["clear", "mist", "light", "heavy"])
    df = pd.get_dummies(df, columns = one_hot_encode, drop_first = True)
    remaining_categories = df.select_dtypes(include = "category").columns
    df[remaining_categories] = df[remaining_categories].apply(lambda s: s.cat.codes + 1 if s.name != "weekday" else s.cat.codes) # All categories are 1-based indexed except weekday which is 0-based

    # Use "dteday" feature to ensure dataset is sorted chronologically, then drop "dteday"
    df["dteday"] = pd.to_datetime(df["dteday"], yearfirst = True)
    df = df.sort_values("dteday", ascending = True)
    df = df.drop("dteday", axis = 1)

    # Get indices of real-valued columns for scaling and generate column name to index dictionary for downstream use
    feature_cols = [c for c in df.columns if c != "cnt"]
    feature_index = {col: i for i, col in enumerate(feature_cols)}
    continuous_cols = ["temp", "hum", "windspeed"]
    continuous_index = [feature_cols.index(col) for col in continuous_cols]

    # Split dataset into training and testing subsets in a time-aware manner due to the forecast-like nature of the task
    split_index = int(len(df) * train_frac)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Separate training and testing sets into X_train, X_test, y_train, and y_test as NumPy arrays
    X_train = train_df[feature_cols].to_numpy().astype(np.float64)
    X_test = test_df[feature_cols].to_numpy().astype(np.float64)
    y_train = train_df["cnt"].to_numpy().astype(np.float64)
    y_test = test_df["cnt"].to_numpy().astype(np.float64)

    # Use OutlierHandler to clip extreme values without losing data points
    handler = OutlierHandler(outlier_method, continuous_index)
    handler.fit(X_train)
    X_train = handler.transform(X_train)

    # Apply the same transformation as above to the testing set
    X_test = handler.transform(X_test)

    # Use ArrayScaler object to standardize the real-valued features in the training set to have mean 0 and variance 1
    scaler = ArrayScaler(scaling_method, continuous_index)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Apply the same transformation as above to the testing set so that they are in the same coordinate system
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_index

class ArrayScaler:

    def __init__(self, method = "zscore", idx = None, eps = 1e-12):

        if idx is None:

            raise ValueError("You must provide idx = [...] to specify which columns to scale.")

        self.method = method
        self.idx = list(idx)
        self.eps = eps
        self.params_ = None

    def fit(self, X_train):

        if self.method == "none":

            self.params_ = None
            return self

        Xs = X_train[:, self.idx].astype(float)

        if self.method == "zscore":

            mu = Xs.mean(axis = 0)
            sigma = Xs.std(axis = 0)
            sigma[sigma < self.eps] = 1.0
            self.params_ = ("zscore", mu, sigma)

        elif self.method == "minmax":

            mn = Xs.min(axis = 0)
            mx = Xs.max(axis = 0)
            denom = mx - mn
            denom[denom < self.eps] = 1.0
            self.params_ = ("minmax", mn, denom)

        elif self.method == "robust":

            med = np.median(Xs, axis = 0)
            q25 = np.percentile(Xs, 25, axis = 0)
            q75 = np.percentile(Xs, 75, axis = 0)
            iqr = q75 - q25
            iqr[iqr < self.eps] = 1.0
            self.params_ = ("robust", med, iqr)

        else:

            raise ValueError(f"Unknown scaling method: {self.method}")

        return self

    def transform(self, X):

        if self.method == "none":
            
            return X

        if self.params_ is None:

            raise ValueError("Scaler is not fitted; please call fit(X_train) first.")

        kind, a, b = self.params_
        out = X.astype(float).copy()
        Xs = out[:, self.idx]

        out[:, self.idx] = (Xs - a) / b

        return out
    
class OutlierHandler:

    def __init__(self, method = "quantile", idx = None, q_low = 0.01, q_high = 0.99, k = 1.5):

        if idx is None:

            raise ValueError("You must provide idx = [...] to specify which columns to handle outliers.")
        
        self.method = method
        self.idx = list(idx)
        self.q_low = q_low
        self.q_high = q_high
        self.k = k
        self.bounds_ = None

    def fit(self, X):

        if self.method == "none":

            self.bounds_ = None
            return self

        Xs = X[:, self.idx].astype(float)

        if self.method == "quantile":

            lo = np.quantile(Xs, self.q_low, axis = 0)
            hi = np.quantile(Xs, self.q_high, axis = 0)

        elif self.method == "iqr":

            q1 = np.quantile(Xs, 0.25, axis = 0)
            q3 = np.quantile(Xs, 0.75, axis = 0)
            iqr = q3 - q1
            lo = q1 - self.k * iqr
            hi = q3 + self.k * iqr

        else:

            raise ValueError(f"Unknown outlier method: {self.method}")

        self.bounds_ = (lo, hi)
        return self

    def transform(self, X):

        if self.method == "none":

            return X

        if self.bounds_ is None:

            raise ValueError("Outlier handler has not been fitted; please call fit(X_train) first.")

        lo, hi = self.bounds_
        out = X.astype(float).copy()
        out[:, self.idx] = np.clip(out[:, self.idx], lo, hi)
        return out

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = process_csv("data/day.csv")

    print("Dataset succesfully loaded and preprocessed with the following shapes for X_train, X_test, y_train, and y_test, respectively:")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("\nIn order to make use of these objects, please import the process_csv(csv_path, train_frac) method from the data_parser module.")
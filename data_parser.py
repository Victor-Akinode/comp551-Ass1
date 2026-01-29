import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_csv(csv_path, train_frac = 0.8):
    """
    Takes path to day.csv from the UCI bike sharing dataset and returns (X_train, X_test, y_train, y_test) as NumPy arrays.

    This implementations drops the following features: ["instant", "casual", "registered", "workingday", "atemp", "dteday"] due
    to relevancy, data leakage, and potential for colinearity. A full justification is provided in the complementary writeup.pdf.

    All non-binary categorical features are one-hot encoded and all real-valued features are standardized based on the training
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

    # One-hot encode all categorical features (first converting them to pd.Categorical so that the column names are clearer for debugging purposes)
    df["mnth"] = pd.Categorical(df["mnth"], categories=range(1, 13)).rename_categories(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    df["season"] = pd.Categorical(df["season"], categories=[1, 2, 3, 4]).rename_categories(["winter", "spring", "summer", "fall"])
    df["weekday"] = pd.Categorical(df["weekday"], categories=range(7)).rename_categories(["sun", "mon", "tue", "wed", "thu", "fri", "sat"])
    df["weathersit"] = pd.Categorical(df["weathersit"], categories=[1, 2, 3, 4]).rename_categories(["clear", "mist", "light", "heavy"])
    df = pd.get_dummies(df, columns=["season", "mnth", "weekday", "weathersit"], drop_first = True)

    # Use "dteday" feature to ensure dataset is sorted chronologically, then drop "dteday"
    df["dteday"] = pd.to_datetime(df["dteday"], yearfirst = True)
    df = df.sort_values("dteday", ascending = True)
    df = df.drop("dteday", axis = 1)

    # Split dataset into training and testing subsets in a time-aware manner due to the forecast-like nature of the task
    split_index = int(len(df) * train_frac)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Use StandardScaler object to standardize the real-valued features in the training set to have mean 0 and variance 1
    continuous_cols = ["temp", "hum", "windspeed"]
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])
    train_df[continuous_cols] = scaler.transform(train_df[continuous_cols])

    # Apply the same transformation as above to the testing set so that they are in the same coordinate system
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

    # Separate training and testing sets into X_train, X_test, y_train, and y_test as NumPy arrays
    X_train = train_df.drop("cnt", axis = 1).to_numpy()
    X_test = test_df.drop("cnt", axis = 1).to_numpy()
    y_train = train_df["cnt"].to_numpy()
    y_test = test_df["cnt"].to_numpy()

    # Introduce bias column to X_train and X_test
    X_train = np.c_[np.ones(len(X_train)), X_train]
    X_test = np.c_[np.ones(len(X_test)), X_test]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = process_csv("day.csv")

    print("Dataset succesfully loaded and preprocessed with the following shapes for X_train, X_test, y_train, and y_test, respectively:")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("\nIn order to make use of these objects, please import the process_csv(csv_path, train_frac) method from the data_parser module.")
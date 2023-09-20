import pandas as pd
import pickle
import os
from scripts.settings import *
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error


def split_data(dataframe: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series):
    """
    Split a dataset into feature and target variables.
    :param dataframe: pd.Dataframe, the input DataFrame containing the dataset to be split.
    :param target: str, the name of the target column in the DataFrame.
    :return: tuple (pd.DataFrame, pd.Series), a tuple containing two components: X - pd.DataFrame - the feature
    variables, which includes all columns except the target. y - pd.Series, the target variable, which is a Pandas
    Series containing the values from the target column.
    """
    X = dataframe.drop(target, axis=1)  # features
    y = dataframe[target]  # target
    return X, y


def main():
    # load train and test datasets
    df_train = pd.read_csv('../prepare_data/df_train.csv')
    df_test = pd.read_csv('../prepare_data/df_test.csv')
    target = 'price'
    # create X and y for train and test data
    X_train, y_train = split_data(df_train, target)
    X_test, y_test = split_data(df_test, target)
    # load the trained random forest model
    rf_regressor_opt = pickle.load(open(os.path.join(trained_model_dir, random_forest_file), 'rb'))
    # Predict the target on the test data
    y_pred = rf_regressor_opt.predict(X_test)

    # Calculate evaluation metrics
    median_absolute_err = median_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Print metrics
    print("Median Squared Error:", median_absolute_err, "Mean Absolute Percentage Error:", mape)


if __name__ == '__main__':
    main()

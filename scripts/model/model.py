import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error
import optuna
from functools import partial


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


def objective(trial, X_train, y_train, X_test, y_test) -> float:
    """
    Optimize hyperparameters for a Random Forest Regressor using Optuna library.
    :param trial: Optuna trial object, used for suggesting hyperparameter values.
    :param X_train: pd.DataFrame, the feature variables of the training dataset.
    :param y_train: pd.Series, the target variable of the training dataset.
    :param X_test: pd.DataFrame, the feature variables of the testing dataset.
    :param y_test: pd.Series, the target variable of the testing dataset.
    :return: float, the median absolute error (MAE) of the model's predictions on the test data.
    """
    # Define hyperparameters to search
    random_state = 42
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
        )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    median_absolute_err = median_absolute_error(y_test, y_pred)

    return median_absolute_err


def main():
    # load train and test datasets
    df_train = pd.read_csv('../prepare_data/df_train.csv')
    df_test = pd.read_csv('../prepare_data/df_test.csv')
    target = 'price'
    # create X and y for train and test data
    X_train, y_train = split_data(df_train, target)
    X_test, y_test = split_data(df_test, target)
    direction = 'minimize'
    n_trials = 300
    random_state = 42

    # create a partially modified objective function
    partial_objective = partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    # create a study for optimization
    study = optuna.create_study(direction=direction)
    study.optimize(partial_objective, n_trials=n_trials)
    best_params = study.best_params

    # Create a Random Forest Regressor using the best_parameters from optuna
    rf_regressor_opt = RandomForestRegressor(**best_params, random_state=random_state)

    # Fit the model on the training data
    rf_regressor_opt.fit(X_train, y_train)

    # Predict the target on the test data
    y_pred = rf_regressor_opt.predict(X_test)

    # Calculate evaluation metrics
    median_absolute_err = median_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Print metrics
    print("Median Squared Error:", median_absolute_err, "Mean Absolute Percentage Error:", mape)


if __name__ == '__main__':
    main()

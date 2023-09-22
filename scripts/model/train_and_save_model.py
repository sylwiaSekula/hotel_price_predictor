from functools import partial
import pickle
import optuna
import os
import pandas as pd
from scripts.settings import *
from scripts.utils import create_dir
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


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


def objective(trial, X_train: pd.DataFrame, y_train: pd.Series, X_test:pd.DataFrame, y_test: pd.Series) -> float:
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

    # create and fit the XGBoost regression model with the suggested hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
        )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    median_absolute_err = median_absolute_error(y_test, y_pred)

    return median_absolute_err


def objective_xgb(trial, X_train: pd.DataFrame, y_train: pd.Series, X_test:pd.DataFrame, y_test: pd.Series) -> float:
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
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'eta': trial.suggest_float('eta', 0.01, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'seed': 42
    }

    # create and fit the XGBoost regression model with the suggested hyperparameters
    model_xgb = xgb.XGBRegressor(**params)
    model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Make predictions on the test set
    y_pred = model_xgb.predict(X_test)

    # Calculate the root mean squared error (RMSE)
    median_absolute_err = median_absolute_error(y_test, y_pred)

    # Optimize for minimizing RMSE
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
    # create directory for models
    create_dir(os.path.dirname(os.path.join(trained_model_dir, random_forest_file)))
    # create a partially modified objective function for random forest regressor
    partial_objective = partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    # create a study for optimization
    study = optuna.create_study(direction=direction)
    study.optimize(partial_objective, n_trials=n_trials)
    best_params = study.best_params

    # create a partially modified objective function for xgboost
    partial_objective_xgb = partial(objective_xgb, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    # create a study for optimization
    study_xgb = optuna.create_study(direction=direction)
    study_xgb.optimize(partial_objective_xgb, n_trials=n_trials)
    best_params_xgb = study_xgb.best_params

    # Create a Random Forest Regressor using the best_parameters from optuna
    rf_regressor_opt = RandomForestRegressor(**best_params, random_state=random_state)
    # fit the XGBoost model using the best_parameters from optuna
    xgb_opt = xgb.XGBRegressor(**best_params_xgb)
    # fit the Linear Regression model
    linear_regression = LinearRegression()

    # Define a list of models and their corresponding file names
    models_and_files = [
        (rf_regressor_opt, random_forest_file),
        (xgb_opt, xgboost_file),
        (linear_regression, linear_regression_file)
    ]

    # Fit each model on the training data and save them
    for model, file_name in models_and_files:
        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Save the trained model
        pickle.dump(model, open(os.path.join(trained_model_dir, file_name), 'wb'))


if __name__ == '__main__':
    main()

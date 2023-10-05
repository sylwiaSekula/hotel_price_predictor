from functools import partial
import pickle
import optuna
import os
import pandas as pd
from scripts.settings import *
from scripts.utils import create_dir
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


def objective_rf(trial, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                 random_state: int) -> float:
    """
    Optimize hyperparameters for a Random Forest Regressor using Optuna library.
    :param trial: Optuna trial object, used for suggesting hyperparameter values.
    :param X_train: pd.DataFrame, the feature variables of the training dataset.
    :param y_train: pd.Series, the target variable of the training dataset.
    :param X_val: pd.DataFrame, the feature variables of the validation dataset.
    :param y_val: pd.Series, the target variable of the validation dataset.
    :param random_state:
    :return: float, the median absolute error of the model's predictions on the validation data.
    """
    # Define hyperparameters to search
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)

    # create and fit the XGBoost regression model with the suggested hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
        )
    model.fit(X_train, y_train)
    # Make predictions on the validation set
    y_pred = model.predict(X_val)
    # Calculate the median squared error
    median_absolute_err = median_absolute_error(y_val, y_pred)
    # Optimize for minimizing the median squared error
    return median_absolute_err


def objective_xgb(trial, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Optimize hyperparameters for a XGboost model using Optuna library.
    :param trial: Optuna trial object, used for suggesting hyperparameter values.
    :param X_train: pd.DataFrame, the feature variables of the training dataset.
    :param y_train: pd.Series, the target variable of the training dataset.
    :param X_val: pd.DataFrame, the feature variables of the validation dataset.
    :param y_val: pd.Series, the target variable of the validation dataset.
    :return: float, the median absolute error of the model's predictions on the validation data.
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
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Make predictions on the validation set
    y_pred = model_xgb.predict(X_val)

    # Calculate the median squared error
    median_absolute_err = median_absolute_error(y_val, y_pred)

    # Optimize for minimizing the median squared error
    return median_absolute_err


def objective_knn(trial, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Optimize hyperparameters for a KNN model using Optuna library.
    :param trial: Optuna trial object, used for suggesting hyperparameter values.
    :param X_train: pd.DataFrame, the feature variables of the training dataset.
    :param y_train: pd.Series, the target variable of the training dataset.
    :param X_val: pd.DataFrame, the feature variables of the validation dataset.
    :param y_val: pd.Series, the target variable of the validation dataset.
    :return: float, the median absolute error of the model's predictions on the validation data.
    """

    # Define hyperparameters to optimize
    n_neighbors = trial.suggest_int('n_neighbors', 1, 20)  # Number of neighbors to consider
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])  # Weighting scheme
    p = trial.suggest_int('p', 1, 2)  # Minkowski distance parameter

    # Create and fit the KNN regression model with the suggested hyperparameters
    model_knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    model_knn.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model_knn.predict(X_val)

    # Calculate the median squared error
    median_absolute_err = median_absolute_error(y_val, y_pred)

    # Optimize for minimizing the median squared error
    return median_absolute_err


def fill_missing_with_knn_train(train_data: pd.DataFrame, columns_to_impute: list,
                                n_neighbors: int) -> pd.DataFrame:
    """
    Impute missing values in train dataset using K-Nearest Neighbors (KNN) imputation with predicted values.
    :param train_data: pd.DataFrame, the training dataset with missing values to be imputed.
    :param columns_to_impute: list, the list of column names to impute missing values for.
    :param n_neighbors: int, the number of neighbors to consider in KNN imputation.
    :return: pd.DataFrame, the imputed training dataset
    """
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    train_data[columns_to_impute] = knn_imputer.fit_transform(train_data[columns_to_impute])
    pickle.dump(knn_imputer, open(os.path.join(trained_model_dir, knn_imputer_file), 'wb'))
    return train_data


def handle_outliers_train(train_data: pd.DataFrame, column_name: str, contamination: float,
                          random_state: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Handle outliers in a DataFrame column using Isolation Forest and replace outliers in a specified column of a
    DataFrame with the median value among the non-outliers.
    :param train_data: pd.DataFrame, the train dataset with outliers
    :param column_name: str: the name of the column with outliers to be handled.
    :param contamination: float, optional: the proportion of outliers in the data.
    :param random_state: int, optional: seed for reproducible results.
    :return:pd.DataFrame, the train dataset with outliers replaced by the median non-outlier
    value in the specified column.
    """
    column_values_train = train_data[column_name].values.reshape(-1, 1)
    isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outliers_train = isolation_forest.fit_predict(column_values_train)
    outlier_indices = train_data.index[outliers_train == -1]
    non_outliers = train_data.loc[~train_data.index.isin(outlier_indices), column_name]
    median_non_outlier = non_outliers.median()
    train_data.loc[outlier_indices, column_name] = median_non_outlier
    pickle.dump(isolation_forest, open(os.path.join(trained_model_dir, isolation_forest_file), 'wb'))
    return train_data


def main():
    random_state = 42
    columns_to_impute = ['rating_score', 'reviews']
    columns_to_encode = ['room_type', 'city']
    output_processed_train_path = ('../prepare_data/df_processed_train.csv')
    n_neighbors = 9
    contamination = 0.05
    direction = 'minimize'
    n_trials = 300
    target = 'price'
    # load train dataset
    df_train = pd.read_csv('../prepare_data/df_processed_train.csv')
    # fill missing values
    df_train = fill_missing_with_knn_train(df_train, columns_to_impute, n_neighbors)
    # handle outliers
    df_train = handle_outliers_train(df_train, target, contamination, random_state)
    # encode categorical data
    encoder = LabelEncoder()
    for col in columns_to_encode:
        df_train[col] = encoder.fit_transform(df_train[col])
    pickle.dump(encoder, open(os.path.join(trained_model_dir, label_encoder_file), 'wb'))
    # save the processed train dateset for data analysis
    df_train.to_csv(output_processed_train_path, index=False)
    # define X and y
    X = df_train.drop(target, axis=1)
    y = df_train[target]
    # split train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=random_state)
    # scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pickle.dump(scaler, open(os.path.join(trained_model_dir, scaler_file), 'wb'))
    # Convert the scaled features back to a DataFrame
    X_val_scaled = scaler.transform(X_val)
    # create directory for models
    create_dir(os.path.dirname(os.path.join(trained_model_dir, random_forest_file)))
    # create a partially modified objective function for random forest regressor
    partial_objective_rf = partial(objective_rf, X_train=X_train_scaled, y_train=y_train, X_val=X_val_scaled, y_val=y_val, random_state=random_state)
    # create a study for optimization Random Forest
    study = optuna.create_study(direction=direction)
    study.optimize(partial_objective_rf, n_trials=n_trials)
    best_params = study.best_params

    # create a partially modified objective function for xgboost
    partial_objective_xgb = partial(objective_xgb, X_train=X_train_scaled, y_train=y_train, X_val=X_val_scaled,
                                    y_val=y_val)
    # create a study for optimization - XGboost
    study_xgb = optuna.create_study(direction=direction)
    study_xgb.optimize(partial_objective_xgb, n_trials=n_trials)
    best_params_xgb = study_xgb.best_params

    # create a partially modified objective function for KNN
    partial_objective_knn = partial(objective_knn, X_train=X_train_scaled, y_train=y_train, X_val=X_val_scaled,
                                    y_val=y_val)
    # create a study for optimization - KNN
    study_knn = optuna.create_study(direction=direction)
    study_knn.optimize(partial_objective_knn, n_trials=n_trials)
    best_params_knn = study_knn.best_params

    # Create a Random Forest Regressor using the best_parameters from optuna
    rf_regressor_opt = RandomForestRegressor(**best_params, random_state=random_state)
    # fit the XGBoost model using the best_parameters from optuna
    xgb_opt = xgb.XGBRegressor(**best_params_xgb)
    # fit the KNN model
    knn_opt = KNeighborsRegressor(**best_params_knn)

    # Define a list of models and their corresponding file names
    models_and_files = [
        (rf_regressor_opt, random_forest_file),
        (xgb_opt, xgboost_file),
        (knn_opt, knn_file)
    ]

    # Fit each model on the training data and save them
    for model, file_name in models_and_files:
        # Fit the model on the training data
        model.fit(X_train_scaled, y_train)

        # Save the trained model
        pickle.dump(model, open(os.path.join(trained_model_dir, file_name), 'wb'))


if __name__ == '__main__':
    main()

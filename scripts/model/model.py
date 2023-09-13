import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  median_absolute_error, mean_absolute_percentage_error
import optuna


def split_data(dataframe: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series):
    """
    Split a dataset into feature and target variables.
    :param dataframe: pd.Dataframe, the input DataFrame containing the dataset to be split.
    :param target: str, the name of the target column in the DataFrame.
    :return: tuple (pd.DataFrame, pd.Series), a tuple containing two components: X - pd.DataFrame - the feature variables,
     which includes all columns except the target. y - pd.Series, the target variable, which is a Pandas Series containing
     the values from the target column.
    """
    X = dataframe.drop(target, axis=1)  # features
    y = dataframe[target]  # target
    return X, y





def main():
    # load the datasets
    df_train = pd.read_csv('../prepare_data/df_train.csv')
    df_test = pd.read_csv('../prepare_data/df_test.csv')
    target = 'price'
    X_train, y_train = split_data(df_train, target)
    X_test, y_test = split_data(df_test, target)

    #define objective function for optuna study
    def objective(trial):
        # Define hyperparameters to search
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        median_absolute_err = median_absolute_error(y_test, y_pred)

        return median_absolute_err

    # create a study for optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=300)

    best_params = study.best_params

    # Create a Random Forest Regressor using the best_parameters from optuna
    rf_regressor_opt = RandomForestRegressor(**best_params, random_state=42)

    # Fit the model on the training data
    rf_regressor_opt.fit(X_train, y_train)

    # Predict the target on the test data
    y_pred = rf_regressor_opt.predict(X_test)

    # Calculate evaluation metrics
    median_absolute_err = median_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Print metrics
    print("Median Squared Error:", median_absolute_err, mape)


if __name__ == '__main__':
    main()

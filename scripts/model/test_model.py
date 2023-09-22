import pandas as pd
import pickle
import os
from scripts.model.train_and_save_model import split_data
from scripts.settings import *
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_absolute_percentage_error


def evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> (float, float, float):
    median_absolute_err = median_absolute_error(y_test, y_pred)
    mean_absolute_err = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return median_absolute_err, mean_absolute_err, mape


def main():
    # load the test dataset
    df_test = pd.read_csv('../prepare_data/df_test.csv')
    target = 'price'
    X_test, y_test = split_data(df_test, target)
    # load the trained models
    rf_regressor_opt = pickle.load(open(os.path.join(trained_model_dir, random_forest_file), 'rb'))
    xgb_opt = pickle.load(open(os.path.join(trained_model_dir, xgboost_file), 'rb'))
    linear_regression = pickle.load(open(os.path.join(trained_model_dir, linear_regression_file), 'rb'))
    # Predict the target on the test data
    y_pred = rf_regressor_opt.predict(X_test)
    y_pred_xgb = xgb_opt.predict(X_test)
    y_pred_lr = linear_regression.predict(X_test)

    metrics_rf = evaluate_model(y_test, y_pred)
    metrics_xgb = evaluate_model(y_test, y_pred_xgb)
    metrics_lr = evaluate_model(y_test, y_pred_lr)

    #  Print metrics
    print("Random Forest=> Median Squared Error, Mean Absolute Error, Mean Absolute Percentage Error:", metrics_rf)
    print("XGBoost=> Median Squared Error, Mean Absolute Error, Mean Absolute Percentage Error:", metrics_xgb)
    print("Linear Regression=> Median Squared Error, Mean Absolute Error, Mean Absolute Percentage Error:", metrics_lr)


if __name__ == '__main__':
    main()

import pandas as pd
import pickle
import os
from scripts.model.train_and_save_model import split_data
from scripts.settings import *
from sklearn.metrics import median_absolute_error, mean_absolute_percentage_error


def main():
    # load the test dataset
    df_test = pd.read_csv('../prepare_data/df_test.csv')
    target = 'price'
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

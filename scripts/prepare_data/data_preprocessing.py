import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def extract_cities(dataframe: pd.DataFrame, city_list: list) -> pd.DataFrame:
    """
    Extract city names from 'location' column and create a new 'city' column.
    :param dataframe: pd.DataFrame, the DataFrame with the 'location' column.
    :param city_list: list, List of city names to extract from the 'location' column.
    :return:pd.DataFrame, the input DataFrame with a new 'city' column containing extracted city names.
    """
    if 'location' not in dataframe.columns:
        raise ValueError("DataFrame must have a 'location' column.")
    dataframe['city'] = dataframe['location'].str.extract('(' + '|'.join(city_list) + ')', expand=False)
    return dataframe


def convert_to_boolean(value: str) -> int:
    """
    Convert a value to a boolean.
    :param value: str, the value to be converted to a boolean.
    :return: int 0 if the value is missing (NaN), 1 otherwise.
    """
    if pd.isna(value):
        return 0
    else:
        return 1


def convert_to_num(dataframe: pd.DataFrame, col: str, value_to_replace: str) -> pd.DataFrame:
    """
    Remove non-numeric characters from a column and convert it to numeric data type.
    :param dataframe: pd.DataFrame, the DataFrame containing the column to be cleaned and converted.
    :param col: str, the name of the column to be processed
    :return: pd.DataFrame, the input DataFrame with the specified column cleaned and converted to numeric type
    """
    dataframe[col] = pd.to_numeric(dataframe[col].str.replace(value_to_replace, '', regex=True), errors='coerce')
    return dataframe


def km_to_meters(distance: str) -> float:
    """
    Convert distance values from kilometers to meters.
    :param distance: str, the distance value to be converted. Example: '5 km' or '500 m'.
    :return: float, the converted distance value in meters.
    """
    if 'km' in distance:
        return float(distance.split()[0]) * 1000
    else:
        return float(distance.split()[0])


def categorize_rooms(row: str) -> str:
    """
    Categorize room types based on room descriptions.
    :param row: str, the description to be categorized.
    :return: str, the categorized room type based on the keywords found in the description.
   """
    if isinstance(row, str) and 'double room' in row.lower():
        return 'double room'
    elif isinstance(row, str) and 'twin room' in row.lower():
        return 'twin room'
    elif isinstance(row, str) and 'apartment' in row.lower():
        return 'apartment'
    return 'other'


def fill_missing_with_knn(train_data: pd.DataFrame, test_data: pd.DataFrame, columns_to_impute: list,
                          n_neighbors: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Impute missing values in train and test datasets using K-Nearest Neighbors (KNN) imputation with predicted values.
    :param train_data: pd.DataFrame, the training dataset with missing values to be imputed.
    :param test_data: pd.DataFrame, the test dataset with missing values to be imputed.
    :param columns_to_impute: list, the list of column names to impute missing values for.
    :param n_neighbors: int, the number of neighbors to consider in KNN imputation.
    :return: (pd.DataFrame, pd.DataFrame), the imputed training and test datasets.
    """
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    train_data[columns_to_impute] = knn_imputer.fit_transform(train_data[columns_to_impute])
    test_data[columns_to_impute] = knn_imputer.transform(test_data[columns_to_impute])

    return train_data, test_data


def handle_outliers(train_data: pd.DataFrame, test_data: pd.DataFrame, column_name: str, contamination: float,
                    random_state: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Handle outliers in a DataFrame column using Isolation Forest and replace outliers in a specified column of a
    DataFrame with the median value among the non-outliers.
    :param train_data: pd.DataFrame, the train dataset
    :param test_data: pd.DataFrame, the test dataset
    :param column_name: str: the name of the column with outliers to be handled.
    :param contamination: float, optional: the proportion of outliers in the data. (Defaults to 0.05)
    :param random_state: int, optional: seed for reproducible results. (Defaults to 42.)
    :return:(pd.DataFrame, pd.DataFrame), the train and test datasets with outliers replaced by the median non-outlier
    value in the specified column.
    """
    column_values_train = train_data[column_name].values.reshape(-1, 1)
    column_values_test = test_data[column_name].values.reshape(-1, 1)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    outliers_train = clf.fit_predict(column_values_train)
    outliers_test = clf.predict(column_values_test)

    dataframes = [train_data, test_data]
    outliers_sets = [outliers_train, outliers_test]
    for dataframe, outliers_set in zip(dataframes, outliers_sets):
        outlier_indices = dataframe.index[outliers_set == -1]
        non_outliers = dataframe.loc[~dataframe.index.isin(outlier_indices), column_name]
        median_non_outlier = non_outliers.median()
        dataframe.loc[outlier_indices, column_name] = median_non_outlier

    return train_data, test_data


def preprocess(dataframe):
    cities = ['Rome', 'Milan', 'Florence', 'Naples']
    columns_to_boolean = ['free_cancellation', 'breakfast']
    columns_to_drop = ['Hotel name', 'location', 'rating']

    # create a new city column by extracting the city from the "location" column
    dataframe = extract_cities(dataframe, cities)
    # convert to boolean the columns "Free_cancellation" and "breakfast"
    for col in columns_to_boolean:
        dataframe[col] = dataframe[col].apply(convert_to_boolean)
    # convert to numeric the column "price" and "reviews"
    dataframe = convert_to_num(dataframe, 'price', ',')
    dataframe = convert_to_num(dataframe, 'reviews', r'[^0-9.]')
    # convert the column 'distance_to_the_city_center' to the same unit of length (meters)
    dataframe['distance_to_the_city_center'] = dataframe['distance_to_the_city_center'].apply(km_to_meters)
    # narrow the data in the column "room_type" to only 4 categories
    dataframe['room_type'] = dataframe['room_type'].apply(categorize_rooms)
    # drop the redundant columns (with no meaningful data)
    dataframe = dataframe.drop(columns=columns_to_drop, axis=1)

    return dataframe


def main():
    columns_to_encode = ['room_type', 'city']
    columns_to_impute = ['rating_score', 'reviews']
    n_neighbors = 9
    column_name = 'price'
    contamination = 0.05
    random_state = 42
    test_size = 0.33
    encoder = LabelEncoder()
    current_directory = os.getcwd()  # Get the current directory
    input_filename = 'italy_hotels.csv'
    input_path = os.path.join(current_directory, input_filename)
    output_train_path = os.path.join(current_directory, 'df_train.csv')
    output_test_path = os.path.join(current_directory, 'df_test.csv')

    # load the dataset
    df = pd.read_csv(input_path)
    # drop duplicates
    df = df.drop_duplicates(ignore_index=True)
    # split the dataset into train and test
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    # preprocess the train dataset
    df_train = preprocess(df_train)
    # preprocess the test dataset
    df_test = preprocess(df_test)
    # encode the columns with categorical data in the dataframe using one hot encoding
    for col in columns_to_encode:
        df_train[col] = encoder.fit_transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
    # detect outliers and replace them with the median non-outlier value in the specified colum
    df_train, df_test = handle_outliers(df_train, df_test, column_name, contamination, random_state)
    # fill the missing values in the "rating_score" and "reviews" column using KNN imputer
    df_train, df_test = fill_missing_with_knn(df_train, df_test, columns_to_impute, n_neighbors)
    # save the preprocessed train dataset to csv
    df_train.to_csv(output_train_path, index=False)
    # save the preprocessed test dataset to csv
    df_test.to_csv(output_test_path, index=False)


if __name__ == '__main__':
    main()

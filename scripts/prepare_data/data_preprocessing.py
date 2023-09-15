import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest


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


# def encode_categorical(dataframe: pd.DataFrame, cols: list) -> pd.DataFrame:
#     """
#     Encode categorical columns in a DataFrame using LabelEncoder.
#     :param dataframe: pd.DataFrame, the DataFrame containing categorical columns to be encoded.
#     :param cols: list of str, names of the columns to be encoded.
#     :return: pd.DataFrame, the input DataFrame with categorical columns encoded using LabelEncoder.
#     """
#     for col in cols:
#         encoder = LabelEncoder()
#         dataframe[col] = encoder.fit_transform(dataframe[col])
#     return dataframe


def fill_missing_with_knn(dataframe: pd.DataFrame, columns_to_impute: list, n_neighbors: int) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using K-Nearest Neighbors (KNN) imputation with predicted values
    :param dataframe: pd.DataFrame, the DataFrame with missing values to be imputed.
    :param n_neighbors: int, optional, the number of neighbors to consider in KNN imputation.
    :return: pd.DataFrame, the input DataFrame with missing values imputed using KNN imputation.
    """
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = knn_imputer.fit_transform(dataframe[columns_to_impute])
    dataframe[columns_to_impute] = imputed_data
    return dataframe


def handle_outliers(dataframe: pd.DataFrame, column_name: str, contamination: float, random_state: int) -> pd.DataFrame:
    """
    Handle outliers in a DataFrame column using Isolation Forest and replace outliers in a specified column of a
    DataFrame with the maximum value among the non-outliers.
    :param dataframe: pd.DataFrame, the DataFrame containing the data.
    :param column_name: str: the name of the column with outliers to be handled.
    :param contamination: float, optional: the proportion of outliers in the data. (Defaults to 0.05)
    :param random_state: int, optional: seed for reproducible results. (Defaults to 42.)
    :return: pd.DataFrame: the DataFrame with outliers replaced by the highest non-outlier value in the specified column.
    """
    column_values = dataframe[column_name].values.reshape(-1, 1)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = clf.fit_predict(column_values)
    outlier_indices = dataframe.index[outliers == -1]
    non_outliers = dataframe.loc[~dataframe.index.isin(outlier_indices), column_name]
    highest_non_outlier = non_outliers.max()
    dataframe.loc[outlier_indices, column_name] = highest_non_outlier
    return dataframe


def preprocess(dataframe):
    cities = ['Rome', 'Milan', 'Florence', 'Naples']
    columns_to_boolean = ['free_cancellation', 'breakfast']
    columns_to_drop = ['Hotel name', 'location', 'rating']
    columns_to_impute = ['rating_score', 'reviews']
    n_neighbors = 9

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
    # encode the columns with categorical data in the dataframe using one hot encoding
    #dataframe = encode_categorical(dataframe, columns_to_encode)
    # detect outliers and replace them with the highest non-outlier value in the specified colum
    dataframe = handle_outliers(dataframe, 'price', 0.05, 42)
    # fill the missing values in the "rating_score" and "reviews" column using KNN imputer
    dataframe = fill_missing_with_knn(dataframe, columns_to_impute, n_neighbors)

    return dataframe


def main():
    columns_to_encode = ['room_type', 'city']
    encoder = LabelEncoder()
    # load the dataset
    df = pd.read_csv('italy_hotels.csv')
    # drop duplicates
    df = df.drop_duplicates(ignore_index=True)
    # split the dataset into train and test
    df_train, df_test = train_test_split(df, test_size=0.33, random_state=42)
    # preprocess the train dataset
    df_train = preprocess(df_train)
    # preprocess the test dataset
    df_test = preprocess(df_test)
    # encode the columns with categorical data in the dataframe using one hot encoding
    for col in columns_to_encode:
        df_train[col] = encoder.fit_transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col])
    # save the preprocessed train dataset to csv
    df_train.to_csv('df_train.csv', index=False)
    # save the preprocessed test dataset to csv
    df_test.to_csv('df_test.csv', index=False)


if __name__ == '__main__':
    main()

import pandas as pd
from sklearn.impute import KNNImputer


def extract_cities(dataframe: pd.DataFrame, city_list: list) -> pd.DataFrame:
    """
    Extract city names from 'location' column and create a new 'city' column.
    :param dataframe: pandas.DataFrame, the DataFrame with the 'location' column.
    :param city_list: list, List of city names to extract from the 'location' column.
    :return:pandas.DataFrame, the input DataFrame with a new 'city' column containing extracted city names.
    """
    # Check if 'location' column exists in the DataFrame
    if 'location' not in dataframe.columns:
        raise ValueError("DataFrame must have a 'location' column.")
    # Create a new 'city' column with extracted city names
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


def remove_string_and_convert_to_numeric(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Remove non-numeric characters from a column and convert it to numeric data type.
    :param dataframe: pandas.DataFrame, the DataFrame containing the column to be cleaned and converted.
    :param col: str, the name of the column to be processed
    :return: pandas.DataFrame, the input DataFrame with the specified column cleaned and converted to numeric type
    """
    dataframe[col] = pd.to_numeric(dataframe[col].str.replace(r'[^0-9.]', '', regex=True))
    return dataframe


def convert_to_numeric(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convert a column's data to numeric format in a DataFrame.
    :param dataframe: pandas.DataFrame, the DataFrame containing the column to be converted.
    :param col: str, the name of the column to be processed.
    :return: pandas.DataFrame, the input DataFrame with the specified column converted to numeric format.
    """
    if dataframe[col].dtype != 'float64':
        dataframe[col] = pd.to_numeric(dataframe[col].str.replace(',', '', regex=True), errors='coerce')
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
    if isinstance(row, str):
        if 'double room' in row.lower():
            return 'double room'
        elif 'twin room' in row.lower():
            return 'twin room'
        elif 'apartment' in row.lower():
            return 'apartment'
    return 'other'


def encode_categorical_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical data in a DataFrame using one-hot encoding.
    :param dataframe: pd.DataFrame, the DataFrame containing categorical columns to be encoded.
    :return: pd.DataFrame, the input DataFrame with categorical columns encoded using one-hot encoding.
    """
    categorical_columns = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns)
    return dataframe


def fill_missing_data(dataframe: pd.DataFrame, fill_value: any) -> pd.DataFrame:
    """
    Fill missing (NaN) data in a DataFrame with a specified fill value.
    :param dataframe: pd.dataframe the DataFrame containing missing data to be filled.
    :param fill_value: any, the value to use for filling missing data.
    :return: pandas.DataFrame anew DataFrame with missing data replaced by the specified fill value.
    """
    dataframe = dataframe.fillna(fill_value, inplace=True)
    return dataframe


def fill_missing_with_knn(dataframe: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame using K-Nearest Neighbors (KNN) imputation with predicted values
    :param dataframe: pd.DataFrame, the DataFrame with missing values to be imputed.
    :param n_neighbors: int, optional, the number of neighbors to consider in KNN imputation.
    :return: pd.DataFrame, the input DataFrame with missing values imputed using KNN imputation.
    """
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = knn_imputer.fit_transform(dataframe)
    dataframe = pd.DataFrame(imputed_data, columns=dataframe.columns)
    return dataframe


def main():
    df = pd.read_csv('italy_hotels.csv')
    cities = ['Rome', 'Milan', 'Florence', 'Naples']
    columns_to_boolean = ['free_cancellation', 'breakfast']
    columns_to_drop = ['Hotel name', 'location', 'rating']

    # drop the duplicates
    df = df.drop_duplicates(ignore_index=True)

    # create a new city column by extracting the city from the "location" column
    extract_cities(df, cities)

    # convert to boolean the columns "Free_cancellation" and "breakfast"
    for col in columns_to_boolean:
        df[col] = df[col].apply(convert_to_boolean)

    # convert to numeric the column "price"
    convert_to_numeric(df, 'price')

    # remove the strings "review" or "reviews" and convert to numeric the column "reviews"
    remove_string_and_convert_to_numeric(df, 'reviews')

    # convert the column 'distance_to_the_city_center' to the same unit of length (meters)
    df['distance_to_the_city_center'] = df['distance_to_the_city_center'].apply(km_to_meters)

    # fill missing values in the column "reviews" with 0 value
    fill_missing_data(df['reviews'], 0)

    # narrow the data in the column "room_type" to only 4 categories
    df['room_type'] = df['room_type'].apply(categorize_rooms)

    # drop the reduntant columns (with meaningful data)
    df = df.drop(columns=columns_to_drop, axis=1)

    # encode categorical data in the dataframe using one hot encoding
    df = encode_categorical_data(df)

    # fill the missing values in the "rating_score" column using KNN imputer
    df = fill_missing_with_knn(df, n_neighbors=9)

    # save the new data frame
    df_processed = df
    df_processed.to_csv('df_processed.csv', index=False)


if __name__ == '__main__':
    main()

import os
import pandas as pd
from sklearn.model_selection import train_test_split


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
    :param value_to_replace:
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
    random_state = 42
    test_size = 0.33
    current_directory = os.getcwd()  # Get the current directory
    input_filename = 'italy_hotels.csv'
    input_path = os.path.join(current_directory, input_filename)
    output_processed_train_path = os.path.join(current_directory, 'df_processed_train.csv')
    output_test_path = os.path.join(current_directory, 'df_test.csv')

    # load the dataset
    df = pd.read_csv(input_path)
    # drop duplicates
    df = df.drop_duplicates(ignore_index=True)
    # split the dataset into train and test
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    # save the test set
    df_test.to_csv(output_test_path, index=False)

    # preprocess the train dataset
    df_train = preprocess(df_train)
    # save the preprocessed train dataset to csv
    df_train.to_csv(output_processed_train_path, index=False)


if __name__ == '__main__':
    main()

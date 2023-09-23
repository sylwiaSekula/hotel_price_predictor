import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_price_distribution(dataframe: pd.DataFrame) -> None:
    """
    Create a box plot to visualize the distribution of 'price' column
    :param dataframe: (pd.DataFrame): the DataFrame containing the data.
    :return: None
    """
    # Create a figure
    plt.figure(figsize=(16, 2))

    # Plot the price column
    sns.boxplot(data=dataframe, x='price', color='#669999')
    plt.title('Distribution of the "Price" column', size=15)

    # Show the plot
    plt.show()


def plot_price_by_city(dataframe: pd.DataFrame) -> None:
    """
    Create a box plot to visualize the distribution of 'price' by 'city'.
    :param dataframe: (pd.DataFrame): the DataFrame containing the data.
    :return: None
    """
    # Create a figure 12x8
    plt.figure(figsize=(12, 8))
    # Plot the boxplot for the price column by city
    sns.boxplot(x='city', y='price', data=dataframe, palette='twilight')
    plt.xlabel('City')
    plt.ylabel('Price')
    # Add a title
    plt.title('Price by City', size=15)
    plt.show()


def plot_rating_score_distribution(dataframe: pd.DataFrame) -> None:
    """
    Create a histogram with a KDE plot to visualize the distribution of 'rating_score'.
    :param dataframe: (pd.DataFrame): the DataFrame containing the data.
    :return: None
    """
    # Create a figure 12x8
    plt.figure(figsize=(12, 8))
    # Plot the histplot for the rating_score column
    sns.histplot(data=dataframe, x='rating_score', kde=True, color='#ac3973', bins=25)
    # Add a title
    plt.title('Rating Score - Distribution', size=10)
    plt.show()


def plot_rating_score_by_city(dataframe: pd.DataFrame) -> None:
    """
    Create a box plot to visualize the 'rating_score' by 'city'.
    :param dataframe: (pd.DataFrame): The DataFrame containing the data.
    :return: None
    """
    # Create a figure
    plt.figure(figsize=(12, 8))
    # Plot the boxplot for the rating_score column by city
    sns.boxplot(x='city', y='rating_score', data=dataframe, palette='twilight')
    plt.xlabel('City')
    # Add a title
    plt.title('Rating Score by City', size=15)
    plt.show()


def plot_entries_by_city(dataframe: pd.DataFrame) -> None:
    """
    Create a count plot to visualize the number of entries in the dataset for each city.
    :param dataframe: (pd.DataFrame): the DataFrame containing the data.
    :return: None
    """
    # Create a figure 12x8
    plt.figure(figsize=(12, 8))
    # Plot the countplot for the city column
    sns.countplot(data=dataframe, x='city', palette='twilight')
    # Add a title
    plt.title('Number of Entries in the Dataset for Each City', size=15)
    plt.show()


def plot_room_type_prices_by_city(dataframe: pd.DataFrame) -> None:
    """
    Create a bar plot to visualize room type prices by city.
    :param dataframe: (pd.DataFrame): the DataFrame containing the data.
    :return: None
    """
    # Create a figure 12x8
    plt.figure(figsize=(12, 8))
    # Plot the barplot for the price column by city and room type
    sns.barplot(data=dataframe, x='city', y='price', hue='room_type', palette='twilight')
    # Add a title
    plt.title('Room Type Prices by City', size=15)
    plt.show()


def plot_distance_to_city_center_by_city(dataframe: pd.DataFrame) -> None:
    """
    Create subplots of histograms to visualize 'distance_to_the_city_center' by city.
    :param dataframe: (pd.DataFrame): the DataFrame containing the data.
    :return: None
    """
    # Create a figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Define a list of unique cities in the dataset
    cities = dataframe['city'].unique()

    # Loop through the cities and plot histograms for each in a separate subplot
    for i, city in enumerate(cities):
        # Filter the data for the current city
        city_data = dataframe[dataframe['city'] == city]

        # Determine the subplot position
        row = i // 2
        col = i % 2

        # Plot the histogram for the current city in the appropriate subplot
        sns.histplot(data=city_data, x='distance_to_the_city_center', kde=True, bins=25, color='#24478f',
                     ax=axes[row, col])
        axes[row, col].set_title(f'Distance to the City Center - {city}', size=12)
        axes[row, col].set_xlabel('Distance')
        axes[row, col].set_ylabel('Frequency')
    plt.suptitle('Distance to the City Center by City', size=22)

    # Adjust the layout
    plt.tight_layout()
    plt.show()


def plot_availability_by_city(dataframe: pd.DataFrame) -> None:
    # Set a color palette for the cities'
    """
    Create subplots to visualize the availability of 'breakfast' and 'free_cancellation' by city.
    :param dataframe: (pd.DataFrame): the DataFrame containing the data.
    :return: None
    """
    # Create a figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Set a color palette for the cities
    city_palette = sns.color_palette('plasma', n_colors=len(dataframe['city'].unique()))

    # Filter the data to include only rows where 'breakfast' and 'free_cancellation' are equal to 1
    breakfast_data = dataframe[dataframe['breakfast'] == 1]
    cancellation_data = dataframe[dataframe['free_cancellation'] == 1]

    # Plot the count plots for 'breakfast' and 'free_cancellation' with hue set to 'city'
    sns.countplot(data=breakfast_data, x='breakfast', palette=city_palette, ax=axes[0], hue='city')
    axes[0].set_title('Breakfast Availability')
    axes[0].set_xlabel('Availability')
    axes[0].set_ylabel('Count')

    sns.countplot(data=cancellation_data, x='free_cancellation', palette=city_palette, ax=axes[1], hue='city')
    axes[1].set_title('Free Cancellation Availability')
    axes[1].set_xlabel('Availability')
    axes[1].set_ylabel('Count')

    # Adjust the layout of subplots and add a legend
    plt.tight_layout()
    axes[0].legend(title='City', loc='upper right')

    # Show the modified chart
    plt.show()


def main():
    df_train_processed = pd.read_csv('../prepare_data/df_train.csv')
    df_test_processed = pd.read_csv('../prepare_data/df_test.csv')
    dfs = (df_train_processed, df_test_processed)
    df_processed = pd.concat(dfs)

    # Define a mapping dictionary to unencode 'city' and "room_type"
    city_mapping = {0: 'Florence', 1: 'Milan', 2: 'Naples', 3: 'Rome'}
    room_mapping = {0: 'apartment', 1: 'double_room', 2: 'other', 3: 'twin_room'}

    # Use the mapping to unencode 'city' column and "room type" column
    df_processed['city'] = df_processed['city'].map(city_mapping)
    df_processed['room_type'] = df_processed['room_type'].map(room_mapping)

    plot_price_distribution(df_processed)
    plot_price_by_city(df_processed)
    plot_entries_by_city(df_processed)
    plot_availability_by_city(df_processed)
    plot_rating_score_distribution(df_processed)
    plot_rating_score_by_city(df_processed)
    plot_room_type_prices_by_city(df_processed)
    plot_distance_to_city_center_by_city(df_processed)


if __name__ == '__main__':
    main()
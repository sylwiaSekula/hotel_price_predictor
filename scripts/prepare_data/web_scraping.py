from bs4 import BeautifulSoup
import pandas as pd
import requests


# Define a function for scraping elements
def scrape_element(hotel: list, element_info: dict) -> str:
    """
    Collect hotels data from
    :param hotel: List of hotels scraped data
    :param element_info: Dictionary with the elements to find in the hotels scraped data
    :return: list with scraped element
    """
    element = hotel.find(element_info['tag'], element_info['attributes'])
    if element:
        if 'text_processing' in element_info and element_info['text_processing'] == 'strip':
            scraped_data = element.text.strip()
        elif 'text_processing' in element_info and element_info['text_processing'] == 'strip_replace':
            scraped_data = element.text.strip().replace(element_info['replace_value'], "")
        else:
            scraped_data = element.text
    else:
        scraped_data = None

    return scraped_data


# define a function for creating a final dataframe
def create_dataframe(data: list, columns: list, dataframes_list: list) -> pd.DataFrame:
    """
    :param data: list with scraped elements
    :param columns: list of columns for dataframe
    :param dataframes_list: list of dataframes
    :return: pd.DataFrame - final dataframe with all the hotels data
    """
    df_iteration = pd.DataFrame(data, columns=columns)
    dataframes_list.append(df_iteration)
    final_df = pd.concat(dataframes_list, ignore_index=True)

    return final_df


def main():
    hotels_data = []
    dataframes_list = []
    max_results = 1000
    final_df = pd.DataFrame()
    columns = ['Hotel name', 'location', 'price', 'rating_score', 'rating',
               'free_cancellation', 'breakfast', 'reviews', 'room_type', 'distance_to_the_city_center']
    urls = [
        'https://www.booking.com/searchresults.html?ss=Rome&ssne=Rome&ssne_untouched=Rome&label=gen173nr-1FCAQoggJCFXNlYXJjaF90cmV2aSBmb3VudGFpbkgxWARotgGIAQGYATG4ARnIAQ_YAQHoAQH4AQOIAgGoAgO4AvuqhaYGwAIB0gIkY2NjMzFlNWQtMWVlZi00ZmU2LTg3MmEtZmRjYjMzMDk0NzZh2AIF4AIB&sid=62c12bd1bfddfc752e1606238ce39472&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-126693&dest_type=city&checkin=2024-02-22&checkout=2024-02-23&group_adults=2&no_rooms=1&group_children=0',
        f'https://www.booking.com/searchresults.html?ss=Milan%2C+Lombardy%2C+Italy&ssne=Rome&ssne_untouched=Rome&label=gen173nr-1FCAQoggJCFXNlYXJjaF90cmV2aSBmb3VudGFpbkgxWARotgGIAQGYATG4ARnIAQ_YAQHoAQH4AQOIAgGoAgO4AvuqhaYGwAIB0gIkY2NjMzFlNWQtMWVlZi00ZmU2LTg3MmEtZmRjYjMzMDk0NzZh2AIF4AIB&sid=62c12bd1bfddfc752e1606238ce39472&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-121726&dest_type=city&ac_position=0&ac_click_type=b&ac_langcode=en&ac_suggestion_list_length=5&search_selected=true&search_pageview_id=72b260bb903b0324&ac_meta=GhA3MmIyNjBiYjkwM2IwMzI0IAAoATICZW46BW1pbGFuQABKAFAA&checkin=2024-02-22&checkout=2024-02-23&group_adults=2&no_rooms=1&group_children=0',
        f'https://www.booking.com/searchresults.html?ss=Florence%2C+Tuscany%2C+Italy&ssne=Milan&ssne_untouched=Milan&label=gen173nr-1FCAQoggJCFXNlYXJjaF90cmV2aSBmb3VudGFpbkgxWARotgGIAQGYATG4ARnIAQ_YAQHoAQH4AQOIAgGoAgO4AvuqhaYGwAIB0gIkY2NjMzFlNWQtMWVlZi00ZmU2LTg3MmEtZmRjYjMzMDk0NzZh2AIF4AIB&sid=62c12bd1bfddfc752e1606238ce39472&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-117543&dest_type=city&ac_position=0&ac_click_type=b&ac_langcode=en&ac_suggestion_list_length=5&search_selected=true&search_pageview_id=342160c0064101b0&ac_meta=GhAzNDIxNjBjMDA2NDEwMWIwIAAoATICZW46CGZsb3JlbmNlQABKAFAA&checkin=2024-02-22&checkout=2024-02-23&group_adults=2&no_rooms=1&group_children=0',
        f'https://www.booking.com/searchresults.html?ss=Naples%2C+Campania%2C+Italy&ssne=Florence&ssne_untouched=Florence&label=gen173nr-1FCAQoggJCFXNlYXJjaF90cmV2aSBmb3VudGFpbkgxWARotgGIAQGYATG4ARnIAQ_YAQHoAQH4AQOIAgGoAgO4AvuqhaYGwAIB0gIkY2NjMzFlNWQtMWVlZi00ZmU2LTg3MmEtZmRjYjMzMDk0NzZh2AIF4AIB&sid=62c12bd1bfddfc752e1606238ce39472&aid=304142&lang=en-us&sb=1&src_elem=sb&src=searchresults&dest_id=-122902&dest_type=city&ac_position=0&ac_click_type=b&ac_langcode=en&ac_suggestion_list_length=5&search_selected=true&search_pageview_id=7952610d3e750091&ac_meta=GhA3OTUyNjEwZDNlNzUwMDkxIAAoATICZW46Bm5hcGxlc0AASgBQAA%3D%3D&checkin=2024-02-22&checkout=2024-02-23&group_adults=2&no_rooms=1&group_children=0'
        ]
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
        'Accept-Language': 'en-US, en;q=0.5'}
    element_info_list = [
        {
            'tag': 'div',
            'attributes': {'data-testid': 'title'},
            'text_processing': 'strip'
        },
        {
            'tag': 'span',
            'attributes': {'data-testid': 'address'},
            'text_processing': 'strip'
        },
        {
            'tag': 'span',
            'attributes': {'data-testid': 'price-and-discounted-price'},
            'text_processing': 'strip_replace',
            'replace_value': '\xa0zł'
        },
        {
            'tag': 'div',
            'attributes': {'class': 'a3b8729ab1 d86cee9b25'},
            'text_processing': 'strip'
        },
        {
            'tag': 'div',
            'attributes': {'class': 'a3b8729ab1 e6208ee469 cb2cbb3ccb'},
            'text_processing': 'strip'
        },
        {
            'tag': 'strong',
            'attributes': {},
            'text_processing': None,
            'string_check': 'Free cancellation'
        },
        {
            'tag': 'span',
            'attributes': {'class': 'e05969d63d'},
            'text_processing': None
        },

        {
            'tag': 'div',
            'attributes': {'class': 'abf093bdfe f45d8e4c32 d935416c47'},
            'text_processing': 'split',
            'split_separator': ' '
        },
        {
            'tag': 'span',
            'attributes': {'class': 'df597226dd'},
            'text_processing': 'strip'
        },
        {
            'tag': 'span',
            'attributes': {'data-testid': 'distance'},
            'text_processing': 'strip_replace',
            'replace_value': 'from center'
        }
    ]

    for url in urls:
        results = 0
        results_step = 25

        while results < max_results:
            full_url = f'{url}&offset={results}'
            response = requests.get(full_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # find all the hotel elements in the HTML document
            hotels = soup.findAll('div', {'data-testid': 'property-card'})

            # loop over the hotel elements and extract the data
            if hotels:
                for hotel in hotels:
                    for element_info in element_info_list:
                        scraped_data = scrape_element(hotel, element_info)
                        hotels_data.append(scraped_data)
                    # add scraped elements to the final dataframe
                    final_df = create_dataframe([hotels_data], columns, dataframes_list)
                    # clean the hotels_data - make it ready for the next iteration
                    hotels_data = []
            results += results_step
    # save the final_df to csv file
    final_df.to_csv('italy_hotels.csv', index=False)


if __name__ == '__main__':
    main()

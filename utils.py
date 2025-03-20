"""Package helpers to run stuff in pocs_helpers."""

import pandas as pd
import requests


def get_word_happiness_labmt():
    """Requests data from https://hedonometer.org/words/labMT-en-v2/ and returns
    a tidy dataframe.

    Notes:
    - language is english version 2 (can be changed--visit the URL above to select
    a language).
    """
    link = "https://hedonometer.org/api/v1/words/?format=json&wordlist__title=labMT-en-v2"
    response = requests.request("GET", link, timeout=45)
    data = response.json()
    json_data = data['objects']
    df = pd.DataFrame(json_data)
    df_tidy = df[['rank', 'word', 'word_english', 'happs', 'stdDev']]

    return df_tidy


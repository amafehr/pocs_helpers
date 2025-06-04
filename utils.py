"""Package helpers to run stuff in pocs_helpers."""

import pandas as pd
import requests


def get_word_happiness_labmt(
        language_version_link: str = "https://hedonometer.org/api/v1/words/?format=json&wordlist__title=labMT-en-v2"
    ) -> pd.DataFrame:
    """Requests data on the happiness scores of words from hedonometer.org and
    returns a tidy dataframe.

    Args:
    language_version_link: url to the labMT language version desired.

    Notes:
    - the language used here is english version 2.
    """
    response = requests.request("GET", language_version_link, timeout=45)
    data = response.json()
    json_data = data['objects']
    df = pd.DataFrame(json_data)
    df_tidy = df[['rank', 'word', 'word_english', 'happs', 'stdDev']]

    return df_tidy

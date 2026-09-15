"""Package helpers to run stuff in pocs_helpers."""

import pandas as pd
import requests

OUSY_PATH = f"../../data/ousiometry_data_augmented.tsv"

# build ousiometer scores
ousy_df = pd.read_csv(OUSY_PATH, sep='\t')
ousy_dict = ousy_df[['word', 'power', 'danger', 'structure']].to_dict(orient='records')
OUSY = {}
for row in ousy_dict:
    OUSY[row['word']] = {'power': row['power'],
                         'danger': row['danger'],
                         'structure': row['structure']
                         }


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





"""Common helper functions for complex systems analses."""
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from scipy import stats

# Download a language version from https://hedonometer.org/words/labMT-en-v1/
# TODO: read this is from the web instead of downlaod
LAB_MT = pd.read_csv('data/labmt.csv')
LAB_MT_DICT = dict(zip(LAB_MT['Word'], LAB_MT['Happiness Score']))


def slice_into_windows(time_series_text_tokens: list, window_size):
    """Slice a list of text tokens into windows according to a window size.
    """
    window_size = int(window_size)
    return [time_series_text_tokens[i:i + window_size]
            for i in range(0, len(time_series_text_tokens), window_size)]


# all np operations did not speed this up--if running on windows, get an
# array of time series happiness scores filtered and run calc_avg_happiness
# on windows
# TODO: update function for that (or add note that in practice, it is much faster (x10) to adapte
# this analysis to run the first part 1 time on the entire book THEN cut into windows
# rather than piping each window into this function)
def calc_avg_happiness(book_df: pd.DataFrame, lens_diff: list) -> float:
    """Calculate the average happiness of a book.

    Args:
    book_df: must have word col.
    lens_diff: amount to subtract or add from 5 for the happiness score lens.

    Notes:
    - cite https://doi.org/10.1007/s10902-009-9150-9
    """
    book_df['word'] = book_df['word'].str.lower()  # makes more matches when made lowercase
    book_df['value'] = book_df['word'].map(LAB_MT_DICT).fillna(0)
    mask = (book_df['value'] > 0) & ((book_df['value'] <= 5 - lens_diff[0]) | (book_df['value'] >= 5 + lens_diff[1]))
    combo_subset = book_df.loc[mask]

    # Calculate the weighted average happiness score
    weighted_sum = (combo_subset['f'] * combo_subset['value']).sum()
    total_frequency = combo_subset['f'].sum()

    # catches divide by 0 error if it exists
    return weighted_sum / total_frequency if total_frequency != 0 else 0


def clean_and_tokenize(long_txt):
    """Clean and tokenize an unprocessed UTF-8 text read from a text file.

    helpful: https://regex101.com/
    """
    # Remove underscores
    long_txt = re.sub(r"\_([^_]+)\_", r"\1", long_txt)

    # Frankenstein
    long_txt = re.sub(r'D--n', 'Damn', long_txt)
    # handle dashes and salutations
    long_txt = re.sub(r'[\u002D\u2013\u2014\u2012\u2015\u2E3A\u2212]', ' DASH ', long_txt)  # all the hyphens, en dashes, em dashes
    long_txt = re.sub(r';—--', ' DASH ', long_txt)    # Semicolon + em dash
    long_txt = re.sub(r'Mr.', 'Mr', long_txt)  # Mr. to Mr
    long_txt = re.sub(r'Mrs.', 'Mrs', long_txt)  # Mrs. to Mrs
    long_txt = re.sub(r'Dr.', 'Dr', long_txt)  # Dr. to Dr
    # any whitespace oddities and standardizing quotes
    long_txt = re.sub(r'\s+', ' ', long_txt)
    long_txt = re.sub(r'\s"', ' " ', long_txt)  # Adds space before opening double quotes
    long_txt = re.sub(r'"', ' " ', long_txt)   # Replaces closing double quotes with space before
    long_txt = re.sub(r'“', ' “ ', long_txt)   # Replaces opening curly double quotes with space before
    long_txt = re.sub(r'”', ' ” ', long_txt)   # Replaces closing curly double quotes with space before
    long_txt = re.sub(r'‘', ' ‘ ', long_txt)   # Replaces opening curly single quotes with space before
    # Replace opening single curly quotes (‘) with space-padded single quote
    long_txt = re.sub(r'‘', " ' ", long_txt)
    # Replace closing single curly quotes (’) with a regular single quote
    long_txt = re.sub(r'’', "'", long_txt)
    # Handle opening quote mark before a letter (leave space between the apostrophe and the letter)
    long_txt = re.sub(r'(\s)\'(\w)', r'\1\' \2', long_txt)
    # Handle closing quote mark after a letter (leave space before the apostrophe)
    long_txt = re.sub(r'(\w)\'(\s)', r'\1 \' \2', long_txt)
    # Split off possession indicator ('s) by adding a space before it
    long_txt = re.sub(r"'s", r" 's", long_txt)
    # Remove any white space at the front of the string
    long_txt = re.sub(r"^\s+", "", long_txt)
    long_txt += "\n"  # Appending a newline

    # Tokenize the rest of the text while parsing into time series
    # this tokenizer parses contractions and other punctuation
    # \w+ matches 1+ preceeding words
    # \$[\d\.]+ matches a dollar sign followed by digits and/or decimal points
    # \S+\' matches any non-whitespace character followed by a single quote (apostrophe)
    # [^\w\s] anything that is not a word character and not whitespace
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+|\$[\d\.]+|\S+\'|[^\w\s]")
    tokens = tokenizer.tokenize(long_txt)
    # replace 'DASH' with '---'
    tokens = [token if token != 'DASH' else '---' for token in tokens]

    return tokens


def calculate_linear_model(x: np.ndarray, y: np.ndarray) -> tuple:
    """Calculate a linear regression model given array-like variables."""
    model = stats.linregress(x, y)
    sd_slope = model.stderr  # standard error of the slope
    r2 = model.rvalue ** 2

    return model, r2, sd_slope  # model.coef provides the slope


def make_df_freq_rank(tokens: list) -> pd.DataFrame:
    """Takes a list of tokens (such as a book) and turns it into a dataframe
    with frequency and rank columns.
    """
    # get word frequencies (much faster than nltk.FreqDist(tokens))
    word_freq = Counter(tokens)
    df = pd.DataFrame(word_freq.items(), columns=["word", "f"])
    df = df.sort_values(by="f", ascending=False).reset_index(drop=True)
    df['rank_ties'] = df['f'].rank(method='average', ascending=False)

    return df


def plot_size_rank(df, color='blue'):
    """Plot size rank."""
    plt.scatter(
        df['log_rank_ties'],
        df['log_size'],
        s=5, marker='o', facecolors='none', edgecolors=color
    )
    plt.xlabel('Log$_{10}$ (rank of words)')
    plt.ylabel('Log$_{10}$ (frequency of words)')
    # plt.title('Size-rank plot')


# TODO:
# heaps law function (take funcs from convo_analyzer)
# yule coefficients of 2 bodies (we did this in Dsci but not sure if it's good practice)
# SVD end-to-end example (matrix-ify, investigate results, visualize top contributes by axis and pole)
# get recs from other people
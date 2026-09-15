"""Common helper functions for complex systems analyses."""
# TODO: more robust documentation

import re
import urllib.request
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from utils import *

# Globals

LAB_MT = get_word_happiness_labmt()
LAB_MT_DICT = dict(zip(LAB_MT['word'], LAB_MT['happs']))


########## Text handling

def get_gutenburg_text(url: str) -> str:
    """Get a text (book) from Project Gutenburg (https://www.gutenberg.org/).

    Args:
    url: must navigate to the .txt version of the book

    Note:
    - Alternatively, copy and paste from that URL into a .txt file and use the
    load_text function.
    """
    # Download the corpus
    response = urllib.request.urlopen(url)
    long_txt = response.read().decode('utf8')
    return long_txt


def read_text_from_file(file_path: str) -> str:
    """Read a text file from its local path.

    Args:
    file_path: the local path where the text file is located.
    """
    long_txt = open(file_path, encoding='utf-8').read()
    return long_txt


def read_text_file_as_list(file_path: str) -> list:
    """Read a local text file from path into a raw time series token list.

    Args:
    file_path: the local path where the text file is located.

    Note:
    - Does not apply advanced token cleaning rules.
    """
    # read in text file as a list
    with open(file_path, 'r') as f:
        text_list = f.read().splitlines()
    return text_list


def clean_and_tokenize(long_txt: str) -> list:
    """Clean and tokenize an unprocessed UTF-8 text read from a text file
    (Ex: a Gutenburg book).

    Args:
    long_txt: a string representing the entire text contents (e.g., of a book).

    Notes:
    - https://regex101.com/ is helpful to check what the regex pattern does.
    - rules should be slightly adapted per text (see Frankenstein example).
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
    # TODO: nltk is not really necessary here--tokenize without this? removes a dependency.
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+|\$[\d\.]+|\S+\'|[^\w\s]")
    tokens = tokenizer.tokenize(long_txt)
    # replace 'DASH' with '---'
    tokens = [token if token != 'DASH' else '---' for token in tokens]

    return tokens


def get_set_of_words(list_of_tokens: list, get_frequencies=False) -> set:
    """Get a set of all unique words in the document.

    Args:
    list_of_tokens:
    get_frequencies: False by default. Set to True to get a Counter object (like
    a dictionary) of the unique set of words and their frequencies.
    """
    word_set = set()
    for token in list_of_tokens:
        word_set.add(token)

    if get_frequencies:
        word_set = Counter(list_of_tokens)

    return word_set


########## Calculations

# all np operations did not speed this up--if running on windows, get an
# array of time series happiness scores filtered and run calc_avg_happiness
# on windows
# TODO: generalize and update function for above (or add note that in practice, it is much faster (x10) to adapt
# this analysis to run the first part 1 time on the entire book THEN cut into windows
# rather than piping each window into this function)
def calc_avg_happiness(book_df: pd.DataFrame, lens_diff: list) -> float:
    """Calculate the average happiness of a book.

    Args:
    book_df: must have 'word' col.
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


def calculate_linear_model(x: np.ndarray, y: np.ndarray) -> tuple:
    """Calculate a linear regression model given array-like variables.

    Args:
    x: a variable in numpy array format.
    y: a variable in numpy array format.
    """
    model = stats.linregress(x, y)
    sd_slope = model.stderr  # standard error of the slope
    r2 = model.rvalue ** 2

    return model, r2, sd_slope  # model.slope provides the slope/coefficient


def make_df_freq_rank(tokens: list) -> pd.DataFrame:
    """Takes tokens (such as a book) and makes a dataframe
    with frequency and rank columns.

    Args:
    tokens: a list of tokens (e.g., words in a book's text data).
    """
    # get word frequencies (much faster than nltk.FreqDist(tokens))
    word_freq = Counter(tokens)
    df = pd.DataFrame(word_freq.items(), columns=["word", "f"])
    df = df.sort_values(by="f", ascending=False).reset_index(drop=True)
    df['rank_ties'] = df['f'].rank(method='average', ascending=False)

    return df


def heaps_from_text(list_of_tokens: list, get_word_set=False) -> tuple:
    """Using a list of token strings, produce the total and unique vocabulary
    needed to calculates Heaps' law scaling over a document.
    """
    # Tokenization already happened as part of cleaning
    # hold counts of total words and unique words
    total_words = np.arange(1, len(list_of_tokens) + 1)
    unique_num_words = np.zeros(len(list_of_tokens))
    # Count unique words while progressing through the text
    word_set = set()
    for i, token in enumerate(list_of_tokens):
        word_set.add(token)
        unique_num_words[i] = len(word_set)

    if get_word_set:
        return total_words, unique_num_words, word_set

    return total_words, unique_num_words


def shifting_window_calc(data: np.array, window_size: int) -> np.array:
    """Calculate a sliding window average over some vector of data.

    Notes:
    - Uses numpy's sliding_window_view for efficiency. When manually calculated,
    it is much slower.
    """
    N = len(data)
    num_windows = N - window_size + 1

    results = np.zeroes(num_windows)
    windows = sliding_window_view(data, window_shape=window_size)
    for i in range(num_windows):
        results[i] = np.nanmean(windows[i])
    return results


def fill_ousy(row: dict) -> str:
    """Finds an ousiometer dictionary score for a given word.

    Note: One can use a POS processor like Stanza to retrieve lemmas. Using lemmas,
    one can map either the word or lemma to ousiometer dict words to increase the
    proportion of matches in the dict.
    """
    current_ousy = OUSY.get(row['text'], 0)
    # if not row['lemma_same'] and current_ousy == 0:
    #     # print('no PSD and lemma IS diff')
    #     return OUSY.get(row['lemma'], 0)
    return current_ousy


def analyze_ousiometry(df: pd.DataFrame) -> pd.DataFrame:
    """Maps all given words to ousiometer scores.
    """
    df['pds'] = df.apply(lambda row: fill_ousy(row), axis=1)
    df['pds'] = df['pds'].replace(0, np.nan)
    # this expands the pds dict to columns
    df = df.join(pd.json_normalize(df['pds']))
    return df


########## Temporal measures: largely influenced by Goh & Barbasi (2008) and Altmann et al. (2009)


def diffs_total_tokens(tokens):
    """
    Return the inter-arrival gaps. For each token, you get a list of distances
    between appearances. This is useful on whole documents or documented subsetted
    to a category (such as part-of-speech).
    """
    unique_tokens = set(tokens)
    counts = {tok: [] for tok in unique_tokens}
    last_seen = {}

    for i, tok in enumerate(tokens):
        if tok in last_seen:
            counts[tok].append(i - last_seen[tok])
        last_seen[tok] = i

    return counts


def burstiness_b(data: np.array) -> float:
    """Variation of inter-arrival times over a data series (Goh & Barbasi, 2008;
    see also Altmann et al., 2009).

    B (burstiness) gives a measure of burstiness, with -1 < B < 1.
    B > 0 indicates bursty behavior (nouns), B < 0 indicates more regular behavior
    than random (determiners), and B = 0 indicates a Poisson process.
    """
    mean = np.mean(data)
    sd = np.std(data)
    b = (sd - mean) / (sd + mean)

    return b


def burstiness_per_word(tokens: list) -> dict:
    """
    Compute burstiness per word type in a sequence of tokens.

    Note: words occuring only twice with have 1 interval (sd=0, mu>0) and thus B=-1, so
    rare words will stack. Thus, we set the minimum to <= 2, which does filter
    out rare occurences (<=2 times).
    """
    positions = defaultdict(list)
    for i, tok in enumerate(tokens):
        positions[tok].append(i)

    burstiness_scores = {}
    for word, idxs in positions.items():
        if len(idxs) <= 2:
            continue
        intervals = np.diff(idxs)  # inter-arrival times
        burstiness_scores[word] = burstiness_b(intervals)

    return burstiness_scores


def memory_m(data) -> float:
    """
    Correlation measure of inter-arrival times (Goh & Barbasi, 2008;
    see also Altmann et al., 2009). Tests if gaps
    are independent or show memory.

    M (memory) is a correlation-based signal with -1 < M < 1.
    Positive M indicates long waits tend to be followed by long waits.
    Negative M means alternation between long and short waits.
    M = 0 indicates no memory (Poisson).
    """
    taus = np.array(data)
    if len(taus) < 2:
        return np.nan
    mean = np.mean(taus)
    numerator = np.sum((taus[:-1] - mean) * (taus[1:] - mean))
    denominator = len(taus - 1) * (np.std(taus) ** 2)
    if denominator == 0:
        return np.nan
    m = numerator / denominator
    return m


def coef_of_variation(data):
    """
    Provides the coefficient of variation (CV), useful when applied to
    distributions to get a sense of spread or to see if the distribution (inter-arrival
    times) is Poisson.

    For a Poisson process, CV should be approximately 1 because variance ≈ mean; also
    B ≈ 0 and M ≈ 0.
    """
    mean = np.mean(data)
    sd = np.std(data)
    cv = sd / mean if mean != 0 else np.nan

    return cv


########## Visualization

# TODO: generalize this and change col names
def plot_size_rank(df: pd.DataFrame, color: str = 'blue'):
    """Plot size rank.

    Args:
    df: a dataframe containing the columns 'log_rank_ties' and 'log_size'
    color: optional (set to blue by default).
    """
    plt.scatter(
        df['log_rank_ties'],
        df['log_size'],
        s=5, marker='o', facecolors='none', edgecolors=color
    )
    plt.xlabel('Log$_{10}$ (rank of words)')
    plt.ylabel('Log$_{10}$ (frequency of words)')
    # plt.title('Size-rank plot')



# TODO:
# add some text manipulations
# heaps law function (take funcs from convo_analyzer--current paper in progress)
# add ways we explore/view raw data, such as:
    # Maybe 2-3 gram and cut it off at top 50 phrases showing up
# yule coefficients of 2 bodies (we did this in Dsci but not sure if it's good practice)
# SVD end-to-end example (matrix-ify, investigate results, visualize top contributors by axis and pole)
# distribution exploration (battery of ways to look at)
# CDF
# CCDF
# shifterator plot wrapper?
# JSD to compare distribs

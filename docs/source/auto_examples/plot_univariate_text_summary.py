"""
Univariate Text Summary
=======

Example of univariate eda summary for a text variable

The text summary computes the following:

- Histogram of # of tokens / document
- Histogram of # of characters / document
- Boxplot of # of unique observations of each document
- Countplots for the most common unigrams, bigrams, and trigams
- A table with summary statistics for the text metadata
"""
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import intedact
import nltk

nltk.download("punkt")
nltk.download("stopwords")

# %%
# Here we take a look at the summaries for GDPR violations.
#
data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv",
    sep="\t",
)

table, fig = intedact.text_univariate_summary(data, "summary", fontsize=10)
fig.show()
table

# %%
# By default, the summary does a lot of text cleaning: removing punctuation and stop words, lower casing. We can
# turn all of these off.
#

table, fig = intedact.text_univariate_summary(
    data,
    "summary",
    fontsize=10,
    remove_stop=False,
    remove_punct=False,
    lower_case=False,
)
fig.show()
table

# intedact

Interactive EDA for pandas DataFrames directly in your Jupyer notebook.

# Getting Started 

## Installation

    pip install intedact 

    import nltk

## Univariate EDA

Univariate EDA refers to the process of visualizing and summarizing a single variable.
intedact's univariate EDA allows you to produce summaries of single columns in a pandas dataframe

For interactive univariate EDA simply import the `univariate_eda_interact` function in a jupyter notebook:

    from intedact import univariate_eda_interact
    univarate_eda_interact(df)

Try `univariate_eda_interact` here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattboggess/intedact/HEAD?filepath=demo%2Funivariate_eda_demo.ipynb)

**Supported Summary Types:**
* categorical: Summarize a categorical or low cardinality numerical column
* numeric: Summarize a high cardinality numerical column
* datetime: Summarize a datetime column
* text: Summarize a free form text column
* collection: Summarize a column with collections of values (i.e. lists, tuples, sets, etc.)
* url: Summarize a column containing urls

## Bivariate EDA

Work in progress. Stay tuned!
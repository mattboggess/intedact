# intedact: Interactive EDA

[![Read the Docs](https://readthedocs.org/projects/intedact/badge/?version=latest)](https://intedact.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mattboggess/intedact/blob/master/LICENSE)

Interactive EDA for pandas DataFrames directly in your Jupyter notebook. intedact's goal is to make
common, standardized EDA summaries available with one function call. Instead of copying and pasting the same code
across projects, just use intedact in any initial exploration notebook and explore the dataset right in the notebook.
intedact is interactive so you can adjust these summaries as needed for different datasets, but it strives to
produce the best summaries by default.

Full documentation at [intedact.readthedocs.io](https://intedact.readthedocs.io/en/latest/index.html)

Try intedact here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattboggess/intedact/master?labpath=demo%2Funivariate_eda_demo.ipynb)

# Getting Started

## Installation

Install via pip:

    pip install intedact

Download the following nltk resources for the ngram text summaries.

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

## Univariate EDA

Univariate EDA refers to the process of visualizing and summarizing a single variable.
intedact's univariate EDA allows you to produce summaries of single columns in a pandas dataframe

For interactive univariate EDA simply import the `univariate_eda_interact` function in a jupyter notebook:

    from intedact import univariate_eda_interact
    univarate_eda_interact(
        data, notes_file="optional_file_to_save_notes_to.json", figure_dir="optional_directory_to_save_plots_to"
    )

<img src="https://github.com/mattboggess/intedact/raw/master/demo/univariate_eda_demo.gif"/>

At the top level, one selects the column and the summary type for that column to display. To explore the full dataset,
just toggle through each of the column names. Current supported summary types:

- categorical: Summarize a categorical or low cardinality numerical column
- numeric: Summarize a high cardinality numerical column
- datetime: Summarize a datetime column
- text: Summarize a free form text column
- collection: Summarize a column with collections of values (i.e. lists, tuples, sets, etc.)
- url: Summarize a column containing urls

For each column, one can then adjust parameters for the given summary type to fit your particular dataset. These summaries
try to automatically set good default parameters, but sometimes you need to make adjustments to get the full picture.

See the documentation for [examples](https://intedact.readthedocs.io/en/latest/auto_examples/index.html) of how to statically call the individual univariate summary functions.

## Bivariate EDA

Work in progress. Stay tuned!
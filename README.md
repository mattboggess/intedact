# intedact

Easy, interactive univariate and bivariate EDA for pandas DataFrames.

# Installation

Development version:

    pip install --upgrade git+git://github.com/mattboggess/intedact.git#egg=intedact

# Getting Started

Simply import the `univariate_eda_interact` or `bivariate_eda_interact` functions:

    from intedact import univariate_eda_interact, bivariate_eda_interact

Then simply pass in a dataframe to generate the interactive widgets interface:

    univarate_eda_interact(df)

Try it here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mattboggess/intedact/master?filepath=demo/demo.ipynb)

# Supported Column Types

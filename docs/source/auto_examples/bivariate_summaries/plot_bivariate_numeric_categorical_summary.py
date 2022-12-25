"""
Bivariate Numeric-Categorical Summary
=======

Example of bivariate eda summary for a numeric independent variable and a categorical dependent variable.

The summary computes the following:

- Lineplot with fractions for each level of the categorical variable against quantiles of the numeric variable
"""
import pandas as pd
import plotly

import intedact

# %%
# Here we look at how diamond cut quality changes with carats.
#

data = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
)
fig = intedact.numeric_categorical_summary(
    data, "carat", "cut", num_intervals=5, fig_width=700
)
plotly.io.show(fig)

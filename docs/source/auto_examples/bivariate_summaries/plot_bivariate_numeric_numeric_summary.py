"""
Bivariate Numeric-Numeric Summary
=======

Example of bivariate eda summary for a pair of numeric variables.

The summary computes the following:

- A scatterplot with trend line
- A 2d histogram
- Boxplots of the dependent variable against quantiles of the independent variable
"""
import pandas as pd
import plotly

import intedact

# %%
# Here we take a look at relationship between carat and price in the diamonds dataset
#

data = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
).sample(n=10000)
fig = intedact.numeric_numeric_summary(data, "carat", "price", fig_width=700)
plotly.io.show(fig)

# %%
# By default, it is hard to see much since the distributions are very skewed with outliers. We can tweak
# the plot to actually visualize the distributions in more detail.
#

fig = intedact.numeric_numeric_summary(
    data,
    "carat",
    "price",
    upper_quantile1=0.98,
    hist_bins=100,
    num_intervals=10,
    opacity=0.4,
    fig_width=700,
)
plotly.io.show(fig)

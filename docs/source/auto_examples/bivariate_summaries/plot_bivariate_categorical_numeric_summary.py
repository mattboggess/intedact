"""
Bivariate Categorical-Numeric Summary
=======

Example of bivariate eda summary for a categorical independent variable and a numeric dependent variable.

The summary computes the following:

- Overlapping histogram/kde plots of distributions by level
- Side by side boxplots per level
"""
import pandas as pd
import plotly

import intedact

# %%
# Here we look at how diamond price changes with cut quality
#

data = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
)
data["cut"] = pd.Categorical(
    data["cut"],
    categories=["Fair", "Good", "Very Good", "Premium", "Ideal"],
    ordered=True,
)
fig = intedact.categorical_numeric_summary(data, "cut", "price", fig_width=700)
plotly.io.show(fig)

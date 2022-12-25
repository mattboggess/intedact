"""
Bivariate Categorical-Categorical Summary
=======

Example of bivariate eda summary for a pair of categorical variables

The summary computes the following:

- Categorical heatmap with counts and percentages for each level combo
- Barplot showing distribution of column2's levels within each level of column1
- Lineplot showing distribution of column2's levels across each level of column1
"""
import pandas as pd
import plotly

import intedact

# %%
# Here we look at how diamond cut quality and clarity quality are related.
#

data = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
)
data["cut"] = pd.Categorical(
    data["cut"],
    categories=["Fair", "Good", "Very Good", "Premium", "Ideal"],
    ordered=True,
)
data["clarity"] = pd.Categorical(
    data["clarity"],
    categories=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
    ordered=True,
)
fig = intedact.categorical_categorical_summary(
    data, "clarity", "cut", barmode="group", fig_width=700
)
plotly.io.show(fig)

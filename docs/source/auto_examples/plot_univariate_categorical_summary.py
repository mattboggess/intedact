"""
Univariate Categorical Summary
=======

Example of univariate eda summary for a categorical variable.

The categorical summary computes the following:

- A countplot with counts and percentages by level of the categorical
- A table with summary statistics
"""
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly

import intedact

# %%
# For our first example, we plot the name of countries who have had GDPR violations.
# By default, the plot will try to order and orient the columns appropriately. Here we order by descending count
# and the plot was flipped horizontally due to the number of levels in the variable.
#
data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv",
    sep="\t",
)
fig = intedact.categorical_summary(data, "name", fig_width=700)
plotly.io.show(fig)

# %%
# We can do additional things such as condense extra columns into an "Other" column, add a bar for missing values,
# and change the sort order to sort alphabetically.
#
fig = intedact.categorical_summary(
    data,
    "name",
    include_missing=True,
    order="sorted",
    max_levels=5,
    fig_width=700,
)
plotly.io.show(fig)

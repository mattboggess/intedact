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
import intedact
import seaborn as sns


# %%
# For our first example, we plot the name of countries who have had GDPR violations.
# By default, the plot will try to order and orient the columns appropriately. Here we order by descending count
# and the plot was flipped horizontally due to the number of levels in the variable.
#
data = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv", sep="\t")
table, fig = intedact.categorical_univariate_summary(data, 'name', fontsize=10)
fig.show()
table

# %%
# We can do additional things such as condense extra columns into an "Other" column, add a bar for missing values,
# and change the sort order to sort alphabetically.
#
table, fig = intedact.categorical_univariate_summary(data, 'name', include_missing=True, order="sorted", max_levels=10, fontsize=10)
fig.show()
table

# %%
# To handle ordinal variable sorting, one must convert the column to an ordered categorical data type. Here's an example
# of this for the diamonds dataset.
#

data = sns.load_dataset("diamonds")
data["clarity"] = pd.Categorical(data["clarity"], categories=["I1", "SI1", "SI2", "VS2", "VS1", "VVS2", "VVS1", "IF"], ordered=True)
table, fig = intedact.categorical_univariate_summary(data, "clarity", flip_axis=False, fontsize=10)
fig.show()
table

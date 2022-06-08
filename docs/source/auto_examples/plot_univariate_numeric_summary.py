"""
Univariate Numeric Summary
=======

Example of univariate eda summary for a numeric variable.

The numeric summary computes the following:

- A histogram with optional kde overlay
- A boxplot
- A table with summary statistics
"""
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import intedact
import seaborn as sns

# %%
# Here we take a look at the distribution of carats for the popular diamonds dataset.
#
data = sns.load_dataset("diamonds")
table, fig = intedact.numeric_univariate_summary(data, "carat", fontsize=10)
fig.show()
table

# %%
# Next we take a look at some GDPR violation prices to showcase the other parameters:
#
# - log transformation
# - outlier filtering
# - kde overlay
# - custom bin count

data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv",
    sep="\t",
)
table, fig = intedact.numeric_univariate_summary(
    data, "price", bins=20, kde=True, transform="log", upper_quantile=0.95, fontsize=10
)
fig.show()
table

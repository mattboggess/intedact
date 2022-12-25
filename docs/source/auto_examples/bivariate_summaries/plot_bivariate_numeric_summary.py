"""
Univariate Numeric Summary
=======

Example of univariate eda summary for a numeric variable.

The numeric summary computes the following:

- A histogram
- A boxplot
"""
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly

import intedact

# %%
# Here we take a look at some GDPR violation prices to showcase the other parameters:
#
# - log transformation
# - outlier filtering
# - kde overlay
# - custom bin count

data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv",
    sep="\t",
)
fig = intedact.numeric_summary(
    data, "price", bins=20, transform="log", upper_quantile=0.95, fig_width=700
)
plotly.io.show(fig)

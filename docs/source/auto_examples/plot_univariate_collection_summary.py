"""
Univariate Collection Summary
=======

Example of univariate eda summary for a collection variable (lists, tuples, sets, etc.).

The collection summary computes the following:

- Three separate countplots:
  - Counts for all the unique collections
  - Counts for all the unique entries
  - Counts for the number of entries in each collection
"""
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly

import intedact

# %%
# Here we take a look at which articles of GDPR countries violated. We first have to process the column so it is
# a list and not a string. One can also choose whether to sort the values (ignore order of how they're listed) and
# remove duplicates (only consider unique entries)
#
data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv",
    sep="\t",
)
data["article_violated"] = data["article_violated"].apply(lambda x: x.split("|"))

fig = intedact.collection_summary(data, "article_violated", fig_width=700)
plotly.io.show(fig)

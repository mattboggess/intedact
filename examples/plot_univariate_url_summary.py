"""
Univariate URL Summary
=======

Example of univariate eda summary for an url variable

The URL summary computes the following:

- Countplot for the invididual unique urls
- Countplot for the domains of the urls
- Countplot for the domain suffixes of the urls
- Countplot for the file types of the urls
- A table with summary statistics for the url metadata
"""
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import intedact

# %%
# Here we take a look at the source URL's for countries GDPR violations recordings.
#
data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv",
    sep="\t",
)

table, fig = intedact.url_univariate_summary(data, "source", fontsize=10)
fig.show()
table

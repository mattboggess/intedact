"""
Univariate URL Summary
=======

Example of univariate eda summary for an url variable

The URL summary computes the following:

- Countplot for the invididual unique urls
- Countplot for the domains of the urls
- Countplot for the domain suffixes of the urls
- Countplot for the file types of the urls
"""
import pandas as pd
import plotly

import intedact

# %%
# Here we take a look at the source URL's for countries GDPR violations recordings.
#
data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv",
    sep="\t",
)

fig = intedact.url_summary(data, "source", fig_width=700)
plotly.io.show(fig)

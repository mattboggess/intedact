"""
Univariate Datetime Summary
=======

Example of univariate eda summary for a datetime variable. Here we look at posting times for TidyTuesday tweets.

The datetime summary computes the following:

- A time seriesplot aggregated according to the `ts_freq` parameter
- Barplots showing counts by day of week, month, hour of day, day of month
"""
import warnings

import pandas as pd
import plotly

import intedact

warnings.filterwarnings("ignore")

data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/tidytuesday_tweets/data.csv"
)
data["created_at"] = pd.to_datetime(data.created_at)
fig = intedact.datetime_summary(data, "created_at", fig_width=700)
plotly.io.show(fig)

# %%
# By default, the summary tries to infer reasonable units for the time differences and time series. We can change
# these by using time unit strings for the `ts_freq` and `delta_units` parameters.
#

fig = intedact.datetime_summary(data, "created_at", ts_freq="1 day", fig_width=700)
plotly.io.show(fig)

# %%
# Example of changing plot type, removing trend line, and removing outliers.
#
fig = intedact.datetime_summary(
    data,
    "created_at",
    ts_type="markers",
    trend_line="none",
    upper_quantile=0.99,
    fig_width=700,
)
plotly.io.show(fig)

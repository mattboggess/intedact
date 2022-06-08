"""
Univariate Datetime Summary
=======

Example of univariate eda summary for a datetime variable. Here we look at posting times for TidyTuesday tweets.

The datetime summary computes the following:

- A time seriesplot aggregated according to the `ts_freq` parameter
- A boxplot and histogram of the time differences between successive observations. `delta_units` controls the units of this.
- Barplots showing counts by day of week, month, hour of day, day of month
- A table with summary statistics for the time differences and time series itself
"""
import warnings

import pandas as pd

import intedact

warnings.filterwarnings("ignore")

data = pd.read_csv(
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/tidytuesday_tweets/data.csv"
)
data["created_at"] = pd.to_datetime(data.created_at)
table, fig = intedact.datetime_univariate_summary(data, "created_at", fontsize=10)
fig.show()
table

# %%
# By default, the summary tries to infer reasonable units for the time differences and time series. We can change
# these by using time unit strings for the `ts_freq` and `delta_units` parameters.
#

table, fig = intedact.datetime_univariate_summary(
    data, "created_at", ts_freq="1 day", delta_units="1 minute", fontsize=10
)
fig.show()
table

# %%
# Example of changing plot type, removing trend line, and removing outliers.
#
table, fig = intedact.datetime_univariate_summary(
    data,
    "created_at",
    ts_type="point",
    trend_line=None,
    upper_quantile=0.99,
    fontsize=10,
)
fig.show()
table

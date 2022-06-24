"""
Bivariate Numeric-Numeric Summary
=======

Example of bivariate eda summary for two numeric variables.

The numeric summary computes the following:

- A scatter/hex/histogram/kde 2d plot
- Correlation metrics
"""
import warnings

warnings.filterwarnings("ignore")

import intedact
import seaborn as sns

# %%
# We plot a scatter plot using the famous iris dataset.
#
data = sns.load_dataset("iris")
table, fig = intedact.numeric_numeric_bivariate_summary(
    data, "sepal_length", "petal_length", fontsize=10
)
fig.show()
table

# %%
# Here we show a hex plot instead.
#

table, fig = intedact.numeric_numeric_bivariate_summary(
    data, "sepal_length", "petal_length", plot_type="hex", fontsize=10
)
fig.show()
table

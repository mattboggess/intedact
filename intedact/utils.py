import pandas as pd
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as stats


def iqr(a):
    """
    Calculate the IQR for an array of numbers.
    https://github.com/has2k1/plotnine/blob/bcb93d6cc4ff266565c32a095e40b0127d3d3b7c/plotnine/stats/binning.py
    """
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1


def freedman_diaconis_bins(a, transform='identity'):
    """
    Calculate number of hist bins using Freedman-Diaconis rule.
    https://github.com/has2k1/plotnine/blob/bcb93d6cc4ff266565c32a095e40b0127d3d3b7c/plotnine/stats/binning.py
    """
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if 'log' in transform:
        a = np.log10(a)
    elif transform == 'sqrt':
        a = np.sqrt(a)

    h = 2 * iqr(a) / (len(a) ** (1 / 3))

    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        bins = np.ceil(np.sqrt(a.size))
    else:
        bins = np.ceil((np.nanmax(a) - np.nanmin(a)) / h)

    return min(np.int(bins), 100)


def add_percent_axis(ax, data_size, flip_axis=False):
    """
    Adds a twin axis with percentages to a count axis.
    """
    if flip_axis:
        ax_perc = ax.twiny()
        ax_perc.set_xticks(100 * ax.get_xticks() / data_size)
        ax_perc.set_xlim((100.0 * (float(ax.get_xlim()[0]) / data_size),
                          100.0 * (float(ax.get_xlim()[1]) / data_size)))
        ax_perc.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_perc.xaxis.set_tick_params(labelsize=10)
    else:
        ax_perc = ax.twinx()
        ax_perc.set_yticks(100 * ax.get_yticks() / data_size)
        ax_perc.set_ylim((100.0 * (float(ax.get_ylim()[0]) / data_size),
                          100.0 * (float(ax.get_ylim()[1]) / data_size)))
        ax_perc.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax_perc.yaxis.set_tick_params(labelsize=10)
    ax_perc.grid(False)

    return ax_perc


def preprocess_numeric_variables(data, column1, column2=None, lq1=0, hq1=1, lq2=0, hq2=0, 
                                 transform1='identity', transform2='identity'):
    """
    Utility script for removing lower and upper quantiles of data and preprocessing data
    for particular transformations.
    """
    # remove upper and lower percentiles in case of outliers
    lq1 = data[column1].quantile(lq1)
    hq1 = data[column1].quantile(hq1)
    query_str = f"{column1} >= {lq1} and {column1} <= {hq1}"
    if column2:
        lq2 = data[column2].quantile(lq2)
        hq2 = data[column2].quantile(hq2)
        query_str += f" and {column2} >= {lq2} and {column2} <= {hq2}"
    data = data.query(query_str)
    
    # modify data for transforms 
    if transform1 == 'log_exclude0':
        data = data[data[column1] > 0]
    elif transform1 == 'log':
        data[column1] = data[column1] + 1e-6
        
    if column2:
        if transform2 == 'log_exclude0':
            data = data[data[column2] > 0]
        elif transform2 == 'log':
            data[column2] = data[column2] + 1e-6
    
    return data
    

def categorize_column_type(col_data, discrete_limit):
    from pandas.api.types import is_numeric_dtype
    from pandas.api.types import is_datetime64_any_dtype
    
    if is_datetime64_any_dtype(col_data):
        return 'datetime'
    elif is_numeric_dtype(col_data):
        if len(col_data.unique()) <= discrete_limit:
            return 'discrete_numeric'
        else:
            return 'continuous_numeric'
    elif col_data.dtype.name == 'category':
        if col_data.cat.ordered:
            return 'ordered_categorical'
        else:
            return 'unordered_categorical'
    elif col_data.dtype.name == 'string':
        return 'text'
    elif col_data.dtype.name == 'object':
        test_value = col_data.dropna().iat[0]
        if isinstance(test_value, (list, tuple, set)):
            return 'list'
        elif len(test_value.split(' ')) > 2:
            return 'text (inferred from object)'
        else:
            return 'unordered_categorical (inferred from object)'
    else:
        raise ValueError(f"Unsupported data type {col_data.dtype.name}")


def order_categorical(data, column1, column2=None, level_order='auto', top_n=20, flip_axis=False):
    
    # determine order to plot levels
    if column2:
        value_counts = data.groupby(column1)[column2].median()
    else:
        value_counts = data[column1].value_counts()
        
    if level_order == 'auto':
        if data[column1].dtype.name == 'category':
            if data[column1].cat.ordered:
                order = list(data[column1].cat.categories)
            else:
                order = list(value_counts.sort_values(ascending=False).index)
        else:
            order = sorted(list(value_counts.index))
    elif level_order == 'ascending':
        order = list(value_counts.sort_values(ascending=True).index)
    elif level_order == 'descending':
        order = list(value_counts.sort_values(ascending=False).index)
    elif level_order == 'sorted':
        order = sorted(list(value_counts.index))
    elif level_order == 'random':
        order = list(value_counts.sample(frac=1).index)
    else:
        raise ValueError(f"Unknown level order specification: {level_order}")
        
    # restrict to top_n levels (condense rest into Other)
    num_levels = len(data[column1].unique())
    if num_levels > top_n:
        other_levels = order[top_n - 1:]
        order = order[:top_n] + ['Other']
        if data[column1].dtype.name == 'category':
            data[column1].cat.add_categories(['Other'], inplace=True)
        data[column1][data[column1].isin(other_levels)] = 'Other'
        
    if flip_axis:
        order = order[::-1]
    
    # convert to ordered categorical variable
    data[column1] = pd.Categorical(data[column1], categories=order, ordered=True)
    
    return data[column1]

def fmt_counts(counts, percentages):
    """
    https://plotnine.readthedocs.io/en/stable/tutorials/miscellaneous-show-counts-and-percentages-for-bar-plots.html
    """
    fmt = '{} ({:.1f}%)'.format
    return [fmt(c, p) for c, p in zip(counts, percentages)]

import pandas as pd

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
            return 'sequence'
        elif len(test_value.split(' ')) > 2:
            return 'text (inferred from object)'
        else:
            return 'unordered_categorical (inferred from object)'
    else:
        raise ValueError(f"Unsupported data type {col_data.dtype.name}")

def order_categorical(data, column1, column2, level_order, top_n, flip_axis):
    
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
        print(f"WARNING: {num_levels - top_n} levels condensed into 'Other'")
        other_levels = order[top_n:]
        order = order[:top_n] + ['Other']
        if data[column1].dtype.name == 'category':
            data[column1].cat.add_categories(['Other'], inplace=True)
        data[column1][data[column1].isin(other_levels)] = 'Other'
        
    if flip_axis:
        order = order[::-1]
    
    # convert to ordered categorical variable
    data[column1] = pd.Categorical(data[column1], categories=order, ordered=True)
    
    return data[column1]

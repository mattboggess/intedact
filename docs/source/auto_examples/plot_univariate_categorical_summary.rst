
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_univariate_categorical_summary.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_univariate_categorical_summary.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_univariate_categorical_summary.py:


Univariate Categorical Summary
=======

Example of univariate eda summary for a categorical variable.

The categorical summary computes the following:

- A countplot with counts and percentages by level of the categorical
- A table with summary statistics

.. GENERATED FROM PYTHON SOURCE LINES 12-20

.. code-block:: default

    import warnings

    warnings.filterwarnings("ignore")

    import pandas as pd
    import intedact
    import plotly








.. GENERATED FROM PYTHON SOURCE LINES 21-25

For our first example, we plot the name of countries who have had GDPR violations.
By default, the plot will try to order and orient the columns appropriately. Here we order by descending count
and the plot was flipped horizontally due to the number of levels in the variable.


.. GENERATED FROM PYTHON SOURCE LINES 25-32

.. code-block:: default

    data = pd.read_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv",
        sep="\t",
    )
    fig = intedact.categorical_summary(data, "name", fig_width=700)
    plotly.io.show(fig)




.. raw:: html
    :file: images/sphx_glr_plot_univariate_categorical_summary_001.html





.. GENERATED FROM PYTHON SOURCE LINES 33-36

We can do additional things such as condense extra columns into an "Other" column, add a bar for missing values,
and change the sort order to sort alphabetically.


.. GENERATED FROM PYTHON SOURCE LINES 36-40

.. code-block:: default

    fig = intedact.categorical_summary(
        data, "name", include_missing=True, order="sorted", max_levels=5, fig_width=700,
    )
    plotly.io.show(fig)



.. raw:: html
    :file: images/sphx_glr_plot_univariate_categorical_summary_002.html


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    No missing values for column: name





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.229 seconds)


.. _sphx_glr_download_auto_examples_plot_univariate_categorical_summary.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_univariate_categorical_summary.py <plot_univariate_categorical_summary.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_univariate_categorical_summary.ipynb <plot_univariate_categorical_summary.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

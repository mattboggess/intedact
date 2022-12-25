import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="intedact",
    packages=["intedact"],
    install_requires=[
        "numpy",
        "pandas",
        "plotly",
        "plotnine",
        "tldextract",
        "nltk",
        "ipywidgets",
        "ipython",
        "nbformat",
    ],
    version="0.1.0",
    author="Matthew Boggess",
    author_email="mattboggess7@gmail.com",
    description="Interactive EDA visualizations in your jupyter notebook",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://intedact.readthedocs.io/en/latest/index.html",
    download_url="https://github.com/mattboggess/intedact",
    keywords=[
        "eda",
        "data visualization",
        "data science",
        "pandas",
        "data analysis",
        "python",
    ],
    classifiers=[],
    python_requires=">=3.6",
)

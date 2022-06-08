import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="intedact",
    packages=["intedact"],
    install_requires=[
        "numpy",
        "plotnine==0.7.1",
        "matplotlib",
        "seaborn",
        "tldextract",
        "nltk",
        "ipywidgets",
        "ipython",
        "scikit-misc",
    ],
    version="0.0.1",
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

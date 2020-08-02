import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="intedact", # Replace with your own username
    version="0.0.1",
    author="Matthew Boggess",
    author_email="mattboggess7@gmail.com",
    description="Lighweight, interactive univariate and bivariate EDA visualizations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattboggess/intedact",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

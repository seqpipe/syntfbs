import setuptools


setuptools.setup(
    name="syntfbs",
    version="0.1.0dev",
    author="Lubomir Chorbadjiev",
    author_email="lubomir.chorbadjiev@gmail.com",
    description="Syntetic TFBS dataset generation",
    long_description="Syntetic TFBS dataset generation",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        where=".", exclude=["tests", ],
    ),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

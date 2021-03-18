import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HEMnet",
    version="1.0.0",
    author="Andrew Su, Xiao Tan and Quan Nguyen",
    author_email="a.su@uqconnect.edu.au, xiao.tan@uq.edu.au, quan.nguyen@uq.edu.au",
    description="HEMnet package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BiomedicalMachineLearning/HEMnet",
    project_urls={
        "Bug Tracker": "https://github.com/BiomedicalMachineLearning/HEMnet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "HEMnet"},
    packages=setuptools.find_packages(where="HEMnet"),
    python_requires=">=3.6",
)

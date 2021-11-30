# Import Necessary Files
import setuptools
import re

def read_dependencies():
    """Dependencies in requirements.txt are converted into python list."""
    with open("requirements.txt", "r") as fh:
        dependencies = fh.readlines()
    return dependencies
        
# Load Description Document
with open("README.md", "r") as fh:
    long_description = fh.read()

# Gather Version Information from Python File
with open("electricpy/__init__.py", encoding="utf-8") as fh:
    file_str = fh.read()
    name = re.search('_name_ = \"(.*)\"', file_str).group(1)
    ver = re.search('_version_ = \"(.*)\"', file_str).group(1)
    # Version Breakdown:
    # MAJOR CHANGE . MINOR CHANGE . MICRO CHANGE
    print("Setup for:",name,"   Version:",ver)

# Generate Setup Tools Argument
setuptools.setup(
    name=name,
    version=ver,
    author="Joe Stanley",
    author_email="stan3926@vandals.uidaho.edu",
    description="Electrical Engineering Functions in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/engineerjoe440/ElectricPy",
    packages=setuptools.find_packages(),
    install_requires = read_dependencies(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source Repository": "https://github.com/engineerjoe440/ElectricPy",
        "Bug Tracker": "https://github.com/engineerjoe440/ElectricPy/issues",
        "Documentation": "https://engineerjoe440.github.io/ElectricPy/",
        }
)
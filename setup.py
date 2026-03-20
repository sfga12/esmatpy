from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as reqs:
    install_requires = reqs.read().splitlines()

setup(
    name="esmatpy",
    version="0.1.0",
    author="Sfga",
    description="Python data processing library for the ESMAT project, handling solar wind variables and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sfga12/esmatpy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
)

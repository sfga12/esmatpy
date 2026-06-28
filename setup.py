from setuptools import setup, find_packages, Extension
import os
import sys

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

esmat_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(esmat_root, "src")
cspice_dir = os.path.join(esmat_root, "dependencies", "cspice")
glm_dir = os.path.join(esmat_root, "build", "_deps", "glm-src")

ext_modules = [
    Extension(
        "esmatpy.core",
        ["src/bindings.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            src_dir,
            os.path.join(cspice_dir, "include"),
            glm_dir
        ],
        library_dirs=[
            os.path.join(cspice_dir, "lib")
        ],
        libraries=["cspice"],
        language="c++",
        extra_compile_args=["/std:c++17", "/MT"] if sys.platform == "win32" else ["-std=c++17"],
        extra_link_args=["/NODEFAULTLIB:library"] if sys.platform == "win32" else []
    ),
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as reqs:
    install_requires = reqs.read().splitlines()

if "pybind11" not in install_requires:
    install_requires.append("pybind11")

setup(
    name="esmatpy",
    version="0.1.0",
    author="Sfga",
    description="Python data processing library for the ESMAT project, handling solar wind variables and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

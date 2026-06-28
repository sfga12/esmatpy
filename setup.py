from setuptools import setup, find_packages, Extension
import os
import sys
import urllib.request
import zipfile
import tarfile
import subprocess

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

esmat_root = os.path.abspath(os.path.dirname(__file__))
deps_dir = os.path.join(esmat_root, "deps")
if not os.path.exists(deps_dir):
    os.makedirs(deps_dir)

# Download GLM
glm_dir = os.path.join(deps_dir, "glm")
if not os.path.exists(glm_dir):
    print("Downloading GLM...")
    glm_url = "https://github.com/g-truc/glm/archive/refs/tags/0.9.9.8.zip"
    glm_zip = os.path.join(deps_dir, "glm.zip")
    urllib.request.urlretrieve(glm_url, glm_zip)
    with zipfile.ZipFile(glm_zip, 'r') as zip_ref:
        zip_ref.extractall(deps_dir)
    os.rename(os.path.join(deps_dir, "glm-0.9.9.8"), glm_dir)

# Download CSPICE
cspice_dir = os.path.join(deps_dir, "cspice")
if not os.path.exists(cspice_dir):
    print("Downloading CSPICE...")
    if sys.platform == "win32":
        cspice_url = "https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Windows_VisualC_64bit/packages/cspice.zip"
        cspice_zip = os.path.join(deps_dir, "cspice.zip")
        urllib.request.urlretrieve(cspice_url, cspice_zip)
        with zipfile.ZipFile(cspice_zip, 'r') as zip_ref:
            zip_ref.extractall(deps_dir)
    elif sys.platform == "darwin":
        cspice_url = "https://naif.jpl.nasa.gov/pub/naif/toolkit//C/MacIntel_OSX_AppleC_64bit/packages/cspice.tar.Z"
        cspice_tar = os.path.join(deps_dir, "cspice.tar.Z")
        urllib.request.urlretrieve(cspice_url, cspice_tar)
        subprocess.check_call(["gunzip", "-f", "cspice.tar.Z"], cwd=deps_dir)
        subprocess.check_call(["tar", "-xf", "cspice.tar"], cwd=deps_dir)
        os.rename(os.path.join(deps_dir, "cspice", "lib", "cspice.a"), os.path.join(deps_dir, "cspice", "lib", "libcspice.a"))
    else: # Linux
        cspice_url = "https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z"
        cspice_tar = os.path.join(deps_dir, "cspice.tar.Z")
        urllib.request.urlretrieve(cspice_url, cspice_tar)
        subprocess.check_call(["gunzip", "-f", "cspice.tar.Z"], cwd=deps_dir)
        subprocess.check_call(["tar", "-xf", "cspice.tar"], cwd=deps_dir)
        os.rename(os.path.join(deps_dir, "cspice", "lib", "cspice.a"), os.path.join(deps_dir, "cspice", "lib", "libcspice.a"))

ext_modules = [
    Extension(
        "esmatpy.core",
        ["src/bindings.cpp"],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
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
    description="Python data processing library for the ESMAT project.",
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

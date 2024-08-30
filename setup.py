#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

def get_cupy_package():
    import os
    cuda_version = os.getenv("CUDA_VERSION", "11")
    if cuda_version.startswith("12"):
        return "cupy-cuda12x"
    return "cupy-cuda11x"

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r2>=0.2.7",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "scikit-image",
    "torch",
    "tqdm",
    "aicsimageio", # no longer developed, switch to https://github.com/bioio-devs/bioio
    "antspyx",
    "SimpleITK",
    "opencv-python",
    "colour",
    "cucim", # prevents usage on windows https://github.com/rapidsai/cucim/issues/86
    "open3d",
    get_cupy_package(),
    # "cupy-cuda11x", # this is the package name for cupy/cupyx, depending on cuda version
    # "cupy-cuda12x", # cuda needs to be installed beforehand

    # "cupy",
    # "cupyx",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="Yu Liu",
    author_email="liuyu9671@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="A fusion algorithm in LSFM",
    entry_points={
        "console_scripts": [
            "fuse_illu=FUSE.bin.run_fuse_illu:main",
            "fuse_det=FUSE.bin.run_fuse_det:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="lsfm_fusion",
    name="FUSE",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.9",
    setup_requires=setup_requirements,
    test_suite="lsfm_fuse/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/peng-lab/LSFM-fusion",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.0.1",
    zip_safe=False,
)

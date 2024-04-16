"""setup file for the project."""

# code gratefully take from https://github.com/navdeep-G/setup.py

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
import versioneer
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "hiclass"
DESCRIPTION = "Hierarchical Classification Library."

# URL = 'https://github.com/scikit-learn-contrib/hiclass'
# URL_DOKU = "https://github.com/scikit-learn-contrib/hiclass
URL_GITHUB = "https://github.com/scikit-learn-contrib/hiclass"
URL_ISSUES = "https://github.com/scikit-learn-contrib/hiclass/issues"
EMAIL = "fabio.malchermiranda@hpi.de, Niklas.Koehnecke@student.hpi.uni-potsdam.de"
AUTHOR = "Fabio Malcher Miranda, Niklas Koehnecke"
REQUIRES_PYTHON = ">=3.8,<3.12"
KEYWORDS = ["hierarchical classification"]
DACS_SOFTWARE = "https://gitlab.com/dacs-hpi"
# What packages are required for this module to be executed?
REQUIRED = ["networkx", "numpy", "scikit-learn", "scipy<1.13", "matplotlib"]

# What packages are optional?
# 'fancy feature': ['django'],}
EXTRAS = {
    "ray": ["ray>=1.11.0"],
    "xai": ["shap==0.44.1", "xarray==2023.1.0"],
    "dev": [
        "flake8==4.0.1",
        "pytest==7.1.2",
        "pytest-flake8==1.1.1",
        "pydocstyle==6.1.1",
        "pytest-pydocstyle==2.3.0",
        "pytest-cov==3.0.0",
        "pyfakefs==5.3.5",
        "black==24.2.0",
        "pre-commit==2.20.0",
        "ray",
        "shap==0.44.1",
        "xarray==2023.1.0",
    ],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's _version.py module as a dictionary.
about = {}
# project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
project_slug = "hiclass"
# with open(os.path.join(here, project_slug, '_version.py')) as f:
#    exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Print things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        """Init options."""
        pass

    def finalize_options(self):
        """Finalize method."""
        pass

    def run(self):
        """Run method."""
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    project_urls={
        "Bug Tracker": URL_ISSUES,
        "Source Code": URL_GITHUB,
        # "Documentation": URL_DOKU,
        # "Homepage": URL,
        "Related Software": DACS_SOFTWARE,
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # 'mycli=mymodule:cli'
    entry_points={
        "console_scripts": ["hiclass=hiclass.__main__:main"],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="BSD 3-Clause",
    keywords=KEYWORDS,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
#    # $ setup.py publish support.
#    cmdclass={
#        'upload': UploadCommand,
#    },

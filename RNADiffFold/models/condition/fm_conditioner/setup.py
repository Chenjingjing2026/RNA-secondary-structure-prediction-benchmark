# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup


with open("fm/version.py") as infile:
    exec(infile.read())


setup(
    name="rna-fm",
    version=version,
    description="RNA Foundation Model (rna-fm): Pretrained language models for RNAs. From CUHK AIH Lab.",
    long_description_content_type="text/markdown",
    author="CUHK AIH Lab",
    url="https://github.com/ml4bio/RNA-FM",
    license="MIT",
    packages=["fm", "fm/downstream", "fm/downstream/pairwise_predictor"],
    data_files=[("source_docs/fm", ["LICENSE", "README.md"])],
    zip_safe=True,
    install_requires = [
        'numpy>=1.22.0',
        'pandas>=1.3.1',
        'tqdm>=4.62',
        'scikit-learn>=0.24',
    ],
)

'''


  Attempting uninstall: tqdm
    Found existing installation: tqdm 4.66.5
    Uninstalling tqdm-4.66.5:
      Successfully uninstalled tqdm-4.66.5
  Attempting uninstall: numpy
    Found existing installation: numpy 1.26.3
    Uninstalling numpy-1.26.3:
      Successfully uninstalled numpy-1.26.3
  Attempting uninstall: scipy
    Found existing installation: scipy 1.13.1
    Uninstalling scipy-1.13.1:
      Successfully uninstalled scipy-1.13.1
  Attempting uninstall: pandas
    Found existing installation: pandas 1.5.3
    Uninstalling pandas-1.5.3:
      Successfully uninstalled pandas-1.5.3
  Attempting uninstall: scikit-learn
    Found existing installation: scikit-learn 1.5.2
    Uninstalling scikit-learn-1.5.2:
      Successfully uninstalled scikit-learn-1.5.2
Successfully installed numpy-1.22.0 pandas-1.3.1 rna-fm scikit-learn-0.24.0 scipy-1.11.4 tqdm-4.62.0
'''
# Self aware SGD
![Python](https://badges.aleen42.com/src/python.svg) ![TensorFlow](https://badges.aleen42.com/src/tensorflow.svg) ![conda](https://img.shields.io/badge/%E2%80%8B-conda-%2344A833.svg?style=flat&logo=anaconda&logoColor=44A833) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper "Reliable Incremental Adaptation of Clinical AI Models". Contains a TensorFlow implementation of _Self Aware SGD_, a label-noise robust incremental learning algorithm.

To run experiments, either create a venv with the appropriate requirements and run the notebook from within this:

```python
conda env create -f environment.yml
conda activate env-sgd
```

Alternatively, you can run the notebook directly on Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnshThakur/Self_aware_SGD/blob/main/demo_main.ipynb)

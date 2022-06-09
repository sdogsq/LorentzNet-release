
This folder contains quark-gluon tagging dataset loading modules. We use energyflow [[1]](https://energyflow.network/docs/datasets/) to automatically download and load this dataset. 

We use distributed data loaders and samplers in [`dataset.py`](./dataset.py) to accommodate distributed data-parallel training.

### References
[1] https://energyflow.network/docs/datasets/
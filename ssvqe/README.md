# SSVQE
Minimal working example to recreate [SSVQE paper](https://arxiv.org/abs/1810.09434). Dependencies:
- tensorflow
- tensorflow-quantum
- matplotlib

(Ensure you are choosing matching TF and TFQ versions)

## H2 Experiment

The data for H2 is already generated (and provided). To train the SSVQE on the data, run `python train.py`

## Generate Data

To generate the data for H2 (or other molecules) there are additional dependencies:
- OpenFermion
- OpenFermion-PySCF
- PySCF

Generate the data via `python generate_data.py`

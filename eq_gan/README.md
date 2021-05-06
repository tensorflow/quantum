# EQ-GAN
Minimal working example to recreate [EQ-GAN paper](https://arxiv.org/abs/2105.00080). Run:
`pip install tensorflow`
`pip install tensorflow-quantum`
`pip install gin-config`
`pip install scikit-optimize`
(Ensure you are choosing matching TF and TFQ versions)


If `run_and_save.use_perfect_swap = True`, an exact swap test is implemented for all iterations. If `run_and_save.use_perfect_swap = False`, an exact swap test is implemented for the first half of training, followed by an adversarial swap test with single-qubit Z rotations to correct noise. Running the code results in data with filename `out-perfect_swap-[suffix].npy` or `out-adversarial_swap-[suffix].npy`, where `[suffix]` is set by `run_and_save.save = [suffix]`. Once both files are generated, analyze with `python3 analyze.py [suffix]` to generate plots in `./plots/`.

The configuration files provide parameters already tuned by `python3 run.py --config [config filename] --tune True`.

## Simulated noise experiment

GHZ experiment to learn 1-qubit state |0>+|1> with simulated noise: `python3 run.py --config simulated_noise_experiment_1q.gin`.

GHZ experiment to learn 2-qubit state |00>+|11> with simulated noise: `python3 run.py --config simulated_noise_experiment_1q.gin`.

## Exponential peaks experiment

To run exponential peaks experiment: `python3 run.py --config exp_config.gin` with either `run_and_save.c_type = 'exp0'` or `run_and_save.c_type = 'exp1'` for peaks of class 0 or class 1.

## QNN QRAM experiment

To run the QNN experiment on hardware: `python3 qnn.py`. Output will be in `./qnn_out/`.
# EQ-GAN
Minimal working example to recreate [EQ-GAN paper](https://arxiv.org/abs/2105.00080). Run:
`pip install tensorflow`
`pip install tensorflow-quantum`
`pip install gin-config`
(Ensure you are choosing matching TF and TFQ versions)

`python3 run.py --config=config.gin`

To prepare plots around the data, you will need to modify the `run` function in `train.py`.

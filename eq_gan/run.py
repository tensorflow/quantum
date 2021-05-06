"""Module for running experiments and hyperparam tuning."""
import argparse
import gin
import train

def main(args):
    train.run_and_save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='gin config to execute under.', default='config.gin')
    parser.add_argument(
        '--tune', help='Whether to run hyperparameter tuning.', default=False)
    parser.add_argument(
        '--compare', help='Whether to run model comparison.', default=False)
    args, _ = parser.parse_known_args()

    gin.parse_config_file(args.config)
    main(args)

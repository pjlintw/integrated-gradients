import os
import time
import yaml
import argparse
from pathlib import Path


def get_args():
    """Parse arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="config.yml")

    return parser.parse_args()


def load_config(args):
    """Load configuration file.
    """
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def get_raw_dir(config):
    return os.path.join(os.getcwd(), config['data']['raw_path'], config['data']['dataset'])


def get_processed_dir(config):
    return os.path.join(os.getcwd(), config['data']['processed_path'], config['data']['dataset'])


def get_tfrecord_dir(config):
    return os.path.join(os.getcwd(), config['data']['tfrecord_path'], config['data']['dataset'])


def timeit(method):
    """Measuring execution time of method
    ```python
    @timeit
    def add(x, y):
      return x + y
    >>>f(5, 10)
    f: 0.00095367431640625
    ```
    Args:
        method: callable method
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        ti_dif = (te - ts) * 1000
        print('{}: {:.4f} ms'.format(method.__name__, (ti_dif)))

        return result

    return timed


def create_dir(path):
    """Savely creating recursive directories"""
    if not isinstance(path, Path):
        path = Path(path)

    if not os.path.exists(path):
        os.makedirs(path)
        print('Created directory: {}'.format(path))
    else:
        print('Directory {} already exists.'.format(path))



import importlib
import os
import random


def set_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        numpy = importlib.import_module("numpy")
    except ImportError:
        pass
    else:
        numpy.random.seed(seed)

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

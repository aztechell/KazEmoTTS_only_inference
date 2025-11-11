import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch


_LOGGER = logging.getLogger(__name__)


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = HParams(**value)
            self[key] = value

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)


def get_logger(model_dir: str, filename: str = "train.log") -> logging.Logger:
    """Create a file logger scoped to the given model directory."""
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(path.name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s"
        )
        handler = logging.FileHandler(path / filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    global _LOGGER
    _LOGGER = logger
    return logger


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[object, Optional[object], Optional[float], Optional[int]]:
    """Load a model checkpoint saved during training."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint.get("iteration")
    learning_rate = checkpoint.get("learning_rate")

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    state_dict = checkpoint["model"]
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state_dict, strict=False)

    active_logger = logger or _LOGGER or logging.getLogger(__name__)
    if iteration is not None:
        active_logger.info(
            "Loaded checkpoint '%s' (iteration %s)", checkpoint_path, iteration
        )
    else:
        active_logger.info("Loaded checkpoint '%s'", checkpoint_path)

    return model, optimizer, learning_rate, iteration


def get_hparams_decode(model_dir: Optional[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/train_grad.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=model_dir,
        help="Model name (used to resolve ./logs/<model>)",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234)
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=10,
        help="Number of reverse diffusion steps",
    )
    parser.add_argument(
        "--stoc",
        action="store_true",
        default=False,
        help="Add a stochastic term into decoding",
    )
    parser.add_argument(
        "-g",
        "--guidance",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "-n", "--noise", type=float, default=1.5, help="Noise temperature"
    )
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="Path to a text file"
    )
    parser.add_argument(
        "-r",
        "--generated_path",
        type=str,
        required=True,
        help="Directory for generated audio",
    )

    args = parser.parse_args()
    model_root = Path("./logs") / args.model if args.model else Path("./logs")
    model_root.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    hparams = HParams(**config)
    hparams.model_dir = str(model_root)
    if hasattr(hparams, "train"):
        hparams.train.seed = args.seed

    return hparams, args

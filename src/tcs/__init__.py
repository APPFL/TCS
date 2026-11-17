"""APPFL-native TCS components; register once before constructing agents."""

from .trainer import TCSTrainer
from .aggregator import TCSAggregator

__all__ = ["TCSTrainer", "TCSAggregator", "register"]


def register():
    import appfl.algorithm.trainer
    import appfl.algorithm.aggregator

    appfl.algorithm.trainer.TCSTrainer = TCSTrainer
    appfl.algorithm.aggregator.TCSAggregator = TCSAggregator

import numpy as np
import pytest
from tcs.mnist import MNISTMLP, partition_labels


def test_model_architecture():
    model = MNISTMLP()
    assert [
        (m.in_features, m.out_features) for m in model if hasattr(m, "in_features")
    ] == [(784, 200), (200, 200), (200, 10)]


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.0])
def test_partition_is_disjoint_complete_nonempty_and_reproducible(alpha):
    labels = np.tile(np.arange(10), 100)
    groups = partition_labels(labels, 100, alpha, 42)
    assert all(groups)
    assert sorted(i for g in groups for i in g) == list(range(len(labels)))
    assert groups == partition_labels(labels, 100, alpha, 42)
    assert groups != partition_labels(labels, 100, alpha, 43)

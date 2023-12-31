import numpy as np


def create_row(n: int, classes: np.ndarray) -> np.ndarray:
    """
    Returns a row of timeseries data. The row is a 2D array with shape (n, 3), where n is the number of time steps.
    
    :param n: Number of time steps
    :type n: int
    :param classes: A 1D array representing the types of defects present in the row
    :type classes: np.ndarray

    :return: A row of timeseries data
    :rtype: np.ndarray
    """
    rng = np.random.default_rng()
    base = np.sin(np.linspace((rng.random(3)), (rng.random(3) + np.array([10, 15, 7])), n))
    
    if classes[0] > 0:
        base[rng.integers(0, n), 0] += 2
    if classes[1] > 0:
        base[rng.integers(0, n), 1] -= 2
    if classes[2] > 0:
        x = rng.integers(0, n - 5)
        base[x:x + 4, 2] = 0
    if classes[3] > 0:
        x = rng.integers(0, n - 10)
        base[x:x + 8, 1] += 1.5
    if classes[4] > 0:
        x = rng.integers(0, n - 7)
        base[x:x + 6, 0] += 1.5
        base[x:x + 6, 2] -= 1.5

    base += rng.random(size=base.shape) * .2
    
    return base


def create_dataset(samples: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns a dataset of timeseries data and their corresponding labels.
    
    :param samples: Number of samples to generate
    :type samples: int

    :return: A tuple containing the timeseries data and their corresponding labels
    :rtype: tuple[list[np.ndarray], list[np.ndarray]]
    """
    inputs, outputs = [], []
    rng = np.random.default_rng()

    for _ in range(samples):
        classes = rng.random(5) < .25
        n = rng.integers(40, 60)
        inputs.append(create_row(n, classes))
        outputs.append(classes)

    return inputs, outputs

import numpy as np


if "float" not in np.__dict__:
    np.float = float
if "int" not in np.__dict__:
    np.int = int
if "bool" not in np.__dict__:
    np.bool = bool


from .eval import Evaluator
from . import datasets
from . import metrics
from . import plotting
from . import utils

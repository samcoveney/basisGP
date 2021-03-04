
#import numpy as np

from functools import partial
import jax.numpy as np # NOTE: used jax numpy but called it np
from jax import jit


@partial(jit)
def Linear(X):
    """Linear mean function."""
    H = np.hstack( [np.ones((X.shape[0],1)) , X] )
    return H


@partial(jit)
def Constant(X):
    """Constant mean function."""
    H = np.ones((X.shape[0],1))
    return H



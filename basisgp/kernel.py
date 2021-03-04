
import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jit, jacfwd, grad, value_and_grad, config


#{{{ RBF abstract class
class RBF:
    """ A matrix: RBF kernel, used for other classes that implement a specific basis.
    """

    def __init__(self, dim):
        self.dim = dim
        
    @partial(jit, static_argnums=(0))
    def HP_transform(self, HP): return jnp.log(HP)

    @partial(jit, static_argnums=(0))
    def HP_untransform(self, HP): return jnp.exp(HP)

    def HP_guess(self, num):
        """Constructs a guess (transformation included) """

        low, high = 0.1, 10
        guess = np.random.uniform(self.HP_transform(low), self.HP_transform(high), size = (num, self.dim))
        #guess = self.HP_transform(np.random.uniform(low, high), size = (num, self.dim)) # alternative
        return guess

    @partial(jit, static_argnums=(0))
    def A_matrix(self, lengthscale, xi, xj):
        """A matrix between training data inputs."""

        # NOTE: matrix is symmetric when xi = xj, so doing too much work here... But JAX arrays are immutable, so difficult to used triangular matrix indices

        w = 1.0 / lengthscale
        sqdiff =  ((xi*w)[:,None] -  (xj*w))**2
        expon = jnp.einsum('ijk->ij', sqdiff)
        A = jnp.exp(-0.5*expon)

        return A
 
#}}}


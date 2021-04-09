"""
    MUCM emulator for single outputs.

    Author: Sam Coveney
    Data: 21-04-2020

    TODO:
    * replace the weird MUCM nugget parameterization with a standard one? Why is MUCM inconsistent on this point?
      -> I have done this replacement... I assume it does not affect the ability to integrate out sigma successfully... Jeremy's thesis suggests it's okay

    * from "Uncertainty Analysis and other Inference Tools for Complex Computer Codes":
      "The first is that a(.) can usually only be `computed' subject to observation error. If we can assume that observation errors are normally distributed, then only a simple modification of the theory is needed. The main complication is that the error variance becomes another hyperparameter to be estimated."
    * from "Some Bayesian Numerical Analysis", in the section on smooth, O'Hagan expicitly says that "we simply add Vf (diagonal matrix of noise) to A as defined in 7)"

    NOTE: This means that in this 'integrate out sigma' formulation, we have sigma^2 * (A + nugget * I). In other words, the noise would be sigma^2 * nugget.
          This is an important caveat!

    NOTE: I have replace the sigma^2 estimate from mucm with 1/(n - m - 2) prefactor with Conti et al 1/(n - m) since this gives virtually same result as training sigma explicitly, so I think this is probably more correct...


"""

import numpy as np

import scipy
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import linalg
from scipy.optimize import minimize

from functools import partial
import inspect

import jax.numpy as jnp
from jax import jit, jacfwd, grad, value_and_grad, config
from jax.scipy.linalg import solve_triangular as jst
config.update('jax_enable_x64', True)



#{{{ use '@timeit' to decorate a function for timing
import time
def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        num = 30
        for r in range(num): # calls function 100 times
            result = f(*args, **kw)
        te = time.time()
        print('func: %r took: %2.4f sec for %d runs' % (f.__name__, te-ts, num) )
        return result
    return timed
#}}}


class Emulator:
    """ Emulator class.

        Initialize example (RBF and Linear can be important from this package):

        model = Emulator(kernel = RBF(dim = 5), basis = Linear)

    """


    #{{{ init
    def __init__(self, kernel, basis):

        try:
            if inspect.isclass(kernel): raise ValueError
        except ValueError as e:
            print("[ERROR: ValueError]: 'kernel' must be a kernel class instance, e.g. RBF(dim = 3) not RBF. Exiting.")
            exit()

        self.kernel = kernel 
        self.basis = basis

        # for jitting the likeihood
        # NOTE: To use jax with lbfgs, need to wrap returns to cast to numpy array - https://github.com/google/jax/issues/1510
        new_llh = value_and_grad(self.LLH)
        self.jit_llh = lambda x, y: [ np.array(val) for val in jit(new_llh)(x, y) ]
    #}}}


    # {{{ data handling including scalings
    def set_data(self, x, y):
        """Set data for interpolation, scaled and centred."""

        # save inputs
        if x.ndim == 1: x = x[:,None]
        self.x = x

        # save outputs, centred and scaled
        self.y_mean, self.y_std = np.mean(y), np.std(y)
        #self.y_mean, self.y_std = 0.0, 1.0 # i.e. no scaling
        self.y = self.scale(y)

        # create the H matrix for training data
        self.H_matrix()


    def scale(self, y, stdev = False):
        """Scales and centres y data."""
        offset = 0 if stdev == True else self.y_mean
        return (y - offset) / self.y_std


    def unscale(self, y, stdev = False):
        """Unscales and uncenteres y data into original scale."""
        offset = 0 if stdev == True else self.y_mean

        y = jnp.atleast_1d(y)

        if y.ndim > 1 and stdev == True: # this is for unscaling the variance matrix
            return ( (y * self.y_std**2) )
        else:
            return (y * self.y_std) + offset

    #}}}


    #{{{ regressor basis matrix H
    def H_matrix(self, X = None):

        # this sets the H matrix for training inputs and saves it
        if X is None:
            save_self_H = True
            X = self.x
        else:
            save_self_H = False
        
        # build H matrix
        H = self.basis(X)
        
        if save_self_H:  # save H matrix between training points
            self.H = H
        else:  # return the H matrix for non-training points
            return H
    #}}}


    #{{{ transformations for optmizing nugget
    @partial(jit, static_argnums=(0))
    def nugget_transform(self, x): return jnp.log(x)

    @partial(jit, static_argnums=(0))
    def nugget_untransform(self, x): return jnp.exp(x)
    #}}}


    #{{{ loglikelihood
    @partial(jit, static_argnums=(0, 2))
    def LLH(self, guess, fixed_nugget):
        """
            See Gaussian Processes for Machine Learning, page 29, eq 2.45

            K: the covariance matrix between training data
            A: H K^-1 H.T
            C: K^-1 H.T A^-1 H K^-1
        """

        HP = self.kernel.HP_untransform(guess[0:self.kernel.dim])
         
        # FIXME: I have realized the error now... fixed_nugget is never None, I used to work out settings based on guess length
        #        problem now is that I can't do this because guess length doesn't tell me what is nugget and what is s2 

        if self.train_nugget == False:
            #print("fixing nugget")
            nugget = fixed_nugget
            gn = self.kernel.dim
        else:
            #print("optimizing nugget")
            nugget = self.nugget_untransform(guess[-1])
            gn = self.kernel.dim + 1

        #print("gn:", gn)
        #print("guess shape:", guess.shape)

        y = self.y
        H = self.H
        n, m = self.x.shape[0], self.H.shape[1]

        ## calculate LLH
        if True:

            K = self.kernel.A_matrix(HP, self.x, self.x) + nugget**2 * jnp.eye(self.x.shape[0])
            L = jnp.linalg.cholesky(K)        

            L_y = jst(L, y, lower = True)

            # Q = H A^-1 H
            L_H = jst(L, H, lower = True)
            Q = jnp.dot(L_H.T, L_H)
            LQ = jnp.linalg.cholesky(Q)

            logdetA = 2.0*jnp.sum(jnp.log(jnp.diag(L)))   # log|A|
            logdetQ = 2.0*jnp.sum(jnp.log(jnp.diag(LQ)))  # log|Q| where Q = H K^-1 H.T

            # calculate B = y.T A^-1 H (H A^-1 H)^-1 H A^-1 y
            #                   beta = (H A^-1 H)^-1 H A^-1 y
            tmp = jnp.dot(L_H.T, L_y) # H A^-1 y
            tmp_2 = jst(LQ, tmp, lower = True)
            B = jnp.dot(tmp_2.T, tmp_2)

            if guess.shape[0] > gn:
                #print("non-mucm")
                s2 = self.nugget_untransform(guess[self.kernel.dim])**2
                #llh = 0.5 * ( -jnp.dot(L_y.T, L_y)/s2 + B/s2 - logdetA - logdetQ - (n - m) * jnp.log(2*np.pi) - (n - m) * jnp.log(s2) )
            else:
                #print("mucm")
                s2 = (1.0/(n-m-0))*( jnp.dot(L_y.T, L_y) - B )
                #llh = 0.5*(-(n - m)*jnp.log(s2) - logdetA - logdetQ)

            llh = 0.5 * ( -jnp.dot(L_y.T, L_y)/s2 + B/s2 - logdetA - logdetQ - (n - m) * jnp.log(2*np.pi) - (n - m) * jnp.log(s2) )
            return -llh

#        except jnp.linalg.linalg.LinAlgError as e:
#            print("  WARNING: Matrix not PSD for", guess, ", not fit.")
#            return None
#
#        except ValueError as e:
#            print("  WARNING: Ill-conditioned matrix for", guess, ", not fit.")
#            return None

    #}}}
    

    #{{{ optimization
    def optimize(self, nugget, restarts = 10, mucm = False):
        """Optimize the hyperparameters.
        
           Arguments:
           nugget -- value of the nugget, if None then train nugget, if number then fix nugget
           restart -- how many times to restart the optimizer (default 10).
           mucm -- set True to integrate out sigma^2

        """

        guess = self.kernel.HP_guess(num = restarts)

        # guess for sigma
        if mucm == False:
            sguess = np.random.uniform(self.nugget_transform(1e-1), self.nugget_transform(1e+1), size = restarts).reshape([1,-1]).T  # FIXME: nugget transform for S
            guess = np.append(guess, sguess, axis = 1)

        # nugget stuff
        if nugget is None:  # nugget not supplied; train nugget
            self.train_nugget = True
            fixed_nugget = 0.0
            nguess = np.random.uniform(self.nugget_transform(1e-4), self.nugget_transform(1e-1), size = restarts).reshape([1,-1]).T
            guess = np.append(guess, nguess, axis = 1)
        else:  # nugget supplied; fix nugget
            self.train_nugget = False
            fixed_nugget = np.abs(nugget)


        # construct the header for printing optimzation results
        # FIXME: it would be better to round up to the number of figures we're actually printing
        fmt = lambda x: '+++++' if abs(x) > 1e3 else '-----' if abs(x) < 1e-3 else str("{:1.3f}".format(x))[:5] % x
        hdr = "\033[1mRestart | "
        for d in range( self.kernel.dim): hdr = hdr + " HP{:d} ".format(d) + " | "
        if mucm == False: hdr = hdr + " sig " + " | "
        if nugget is None:  hdr = hdr + " nug " + " | "
        hdr = hdr + "\033[0m"
        print("Optimizing Hyperparameters...\n" + hdr)


        for ng, g in enumerate(guess):
            optFail = False
            #try:
            if True:
                bestStr = "   "
                best = False

                #res = minimize(self.LLH, g, method = 'L-BFGS-B', jac = grad_func) # for jax with separate func and grad 
                res = minimize(self.jit_llh, g, method = 'L-BFGS-B', jac = True, args = (fixed_nugget)) # for jax with joint func and grad
                #res = minimize(self.LLH, g, method = 'L-BFGS-B', jac = False, args = (fixed_nugget)) # for jax with joint func and grad

                if np.isfinite(res.fun):
                    try:
                        if res.fun < bestRes.fun:
                            bestRes = res
                            bestStr = " * "
                            best = True
                    except:
                        bestRes = res
                        bestStr = " * "
                        best = True
                else:
                    bestStr = " ! "

                # FIXME: using HP_untransform for the other parameters too
                prnt = " {:02d}/{:02d} ".format(ng + 1, restarts) + \
                           " | %s" % ' | '.join(map(str, [fmt(i) for i in self.kernel.HP_untransform(res.x)])) + \
                           " | {:s} f: {:.3f}".format(bestStr, np.around(res.fun, decimals=4))
                print(prnt)
                if best: bestPrnt = "\033[1m" + prnt + "\033[0m"

            #except TypeError as e:
            #    optFail = True


        self.store_values(bestRes.x, fixed_nugget) # save other useful things

        print("\n" + bestPrnt)
        print(" (beta:", ", ".join(map(str, [fmt(i) for i in self.beta])), ")")
        if mucm: print(" (mucm sigma: {:f})".format(self.sigma))
        else: print(" (trained sigma: {:f})".format(self.sigma))
        print(" (noise stdev: {:f})".format(self.nugget * self.sigma))
        print("\n")

    #}}}

    
    #{{{ store some important variables
    def store_values(self, guess, fixed_nugget):
        """Calculate and save some important values."""

        # save results of optimization
        # FIXME: save different things based on whether using mucm or not
        
        self.HP = self.kernel.HP_untransform(guess[0:self.kernel.dim])

        if self.train_nugget == False:
            self.nugget = fixed_nugget
            gn = self.kernel.dim
        else:
            self.nugget = self.nugget_untransform(guess[-1])
            gn = self.kernel.dim + 1

        self.A = self.kernel.A_matrix(self.HP, self.x, self.x) + self.nugget**2 * jnp.eye(self.x.shape[0])

        y, H = self.y, self.H
        n, m = self.x.shape[0], self.H.shape[1]

        L = jnp.linalg.cholesky(self.A)        
        L_y = jst(L, y, lower = True)
        L_H = jst(L, H, lower = True)
        Q = jnp.dot(L_H.T, L_H) # Q = H A-1 H
        LQ = jnp.linalg.cholesky(Q)

        tmp = jnp.dot(L_H.T, L_y) # H A^-1 y
        tmp_2 = jst(LQ, tmp, lower = True)
        B = jnp.dot(tmp_2.T, tmp_2) # B = y.T A^-1 H (H A^-1 H)^-1 H A^-1 y

        beta = jst(LQ.T, tmp_2, lower = False) # beta = (H A^-1 H)^-1 H A^-1 y

        if guess.shape[0] > gn:
            s2 = self.nugget_untransform(guess[self.kernel.dim])**2
        else:
            s2 = (1.0/(n-m-0))*( jnp.dot(L_y.T, L_y) - B )


        # save important values
        self.L, self.LQ = L, LQ
        self.L_H = L_H
        self.L_T = jst(L, y - jnp.dot(H, beta), lower = True)
        self.beta = beta
        self.sigma = np.sqrt(s2)

        return
    #}}}


    #{{{ posterior prediction
    @partial(jit, static_argnums=(0,2))
    def posterior(self, X, full_covar):
        """Posterior prediction at points X.
            
           If not returning full covariance matrix, then mean and stdev are returned.

        """

        if X.ndim == 1: X = X[:,None]

        if X.shape[1] != self.x.shape[1]:
            print("[ERROR]: inputs features do not match training data.")
            return

        # load precalculated quantities
        H, y, beta, s2 = self.H, self.y, self.beta, self.sigma**2
        L, LQ = self.L, self.LQ
        L_T, L_H = self.L_T, self.L_H

        # new quantities
        A_cross = self.kernel.A_matrix(self.HP, X, self.x)
        H_pred = self.basis(X)
        L_A_cross = jst(L, A_cross.T, lower = True)
        R = H_pred - jnp.dot(L_A_cross.T, L_H)
        LQ_R = jst(LQ, R.T, lower = True)

        mean = jnp.dot(H_pred, beta) + jnp.dot(L_A_cross.T, L_T) # posterior mean

        if full_covar:

            A_pred = self.kernel.A_matrix(self.HP, X, X)

            tmp_1 = jnp.dot(L_A_cross.T, L_A_cross)
            tmp_2 = jnp.dot(LQ_R.T, LQ_R)

            var = s2 * ( A_pred - tmp_1 + tmp_2 ) # posterior var

            return self.unscale(mean, stdev = False), self.unscale(var, stdev = True) 

        else:

            A_pred = 1.0 

            tmp_1 = jnp.einsum("ij, ji -> i", L_A_cross.T, L_A_cross)
            tmp_2 = jnp.einsum("ij, ji -> i", LQ_R.T, LQ_R)

            var = s2 * ( A_pred - tmp_1 + tmp_2 ) # pointwise posterior var

            return self.unscale(mean, stdev = False), self.unscale(jnp.sqrt(var), stdev = True) 

    #}}}



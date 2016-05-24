import numpy as np
import pylab as plt
import scipy
from numpy.random import permutation
from numpy.linalg import norm
from scipy import rand, randn, ones
from scipy.sparse import csr_matrix

def _rand_sparse(m, n, density):
    # check parameters here
    if density > 1.0 or density < 0.0:
        raise ValueError('density should be between 0 and 1')
    # More checks?
    # Here I use the algorithm suggested by David to avoid ending
    # up with less than m*n*density nonzero elements (with the algorithm
    # provided by Nathan there is a nonzero probability of having duplicate
    # rwo/col pairs).
    nnz = max( min( int(m*n*density), m*n), 0)
    rand_seq = permutation(m*n)[:nnz]
    row  = rand_seq / n
    col  = rand_seq % n
    data = ones(nnz, dtype='int8')
    # duplicate (i,j) entries will be summed together
    # CSR matrix Compressed sparse row matrix
    return csr_matrix( (data,(row,col)), shape=(m,n) )

def sprandn(m, n, density):
    """Build a sparse normally distributed random matrix
       Parameters
       ----------
       m, n     : dimensions of the result (rows, columns)
       density  : fraction of nonzero entries.
       Example
       -------
       >>> from scipy.sparse import sprandn
       >>> print sprandn(2, 4, 0.5).todense()
       matrix([[-0.84041995,  0.        ,  0.        , -0.22398594],
               [-0.664707  ,  0.        ,  0.        , -0.06084135]])
    """
    A = _rand_sparse(m, n, density)
    A.data = randn(A.nnz)
    return A

def prox_l1(v,lamb):
    # np.maximum does elementwise maximum - soft thresholding for L1 norm
    return np.maximum(0, v - lamb) - np.maximum(0, -v - lamb);

def prox_matrix(v, lamb, prox_f):
    """
     Proximal operator of a matrix function
     Suppose F is a orthogonally invariant matrix function such that
     F(X) = f(s(X)), where s is the singular value map and f is some
     absolutely symmetric function. Then

     X = prox_matrix(V,lambda,prox_f)

     evaluates the proximal operator of F via the proximal operator
     of f. Here, it must be possible to evaluate prox_f as prox_f(v,lambda).

     if prox_f = prox_l1 evaluates the proximal operator of the nuclear norm at V
     singular value thresholding
    """
    # full_matrices=True gives U mxm s kxk V nxn
    # full_matrices=False gives U m*K S kxk V kxn
    _U, _s, _V = np.linalg.svd(v, full_matrices=False)
    return np.dot(_U , np.dot( np.diag(prox_f(_s, lamb)) , _V) )

def objective(X1, g2, X2, g3, X3):
    # frob norm on gaussian error, l1 norm on sparse matrix
    _U, singular_values, _V = np.linalg.svd(X3)
    return norm(X1,'fro')**2 + g2*norm(X2.flatten(),1) + g3*norm(singular_values,1)


m = 20
n = 50

N = 3    # Number of decompositions
r = 4    # Rank of the low rank component

L = np.dot( np.random.randn(m,r), np.random.randn(r,n) )  # low rank matrix
S = sprandn(m, n, 0.05).todense()           # sparse matrix

#Sparse matrix with very high error
#np.nonzero returns indices of nonzero elements
#np.random.binomial(1, 0.5) returns either 0 or 1 with equal probability
S[np.nonzero(S)] =  20 * np.random.binomial(1, 0.5, np.count_nonzero(S)) -10

V = 0.001*np.random.randn(m,n)                      # noise

A = S + L + V              # matrix composed of lowrank, sparse, gaussia noise

A_row = np.array(np.hstack(A).tolist()[0])
g2_max = norm(A_row, ord=np.inf)    # max norm across all elements of A
g3_max = norm(A)
g2 = 0.15*g2_max
g3 = 0.15*g3_max

MAX_ITER = 1000
ABSTOL   = 1e-4
RELTOL   = 1e-2

reg = 1.0 #lambda
rho = 1/reg;

X_1 = np.zeros((m, n))
X_2 = np.zeros((m, n))
X_3 = np.zeros((m, n))
z   = np.zeros((m, N*n))    # z has one instantiation for each matrix
U   = np.zeros((m,n))


#print L, S, A
admm_cost = []
for k in range(0,MAX_ITER):

    B = (X_1 + X_2 + X_3)/N - A*(1.0/N) + U

    # x-update
    X_1 = (1.0/(1+reg))*(X_1 - B)    #proximal operator for gaussian noise

    X_2 = prox_l1(X_2 - B, reg*g2);  #proximal operator for sparse matrix constraint

    X_3 = prox_matrix(X_3 - B, reg*g3, prox_l1) #proximal operator for low rank matrix

    #z-update
    x = np.hstack([X_1, X_2, X_3])
    zold = z
    # np.tile repeats a matrix by a size (m,n)
    z = x + np.tile( (A - X_1 - X_2 - X_3)/N, (1, N))

    # u-update
    U = B

    # Termination checks
    admm_cost.append(objective(X_1, g2, X_2, g3, X_3))
    r_norm   = norm(x - z,'fro')
    s_norm   = norm(-rho*(z - zold),'fro')
    eps_pri  = np.sqrt(m*n*N)*ABSTOL + RELTOL*np.maximum(norm(x,'fro'), norm(-z,'fro'))
    eps_dual = np.sqrt(m*n*N)*ABSTOL + RELTOL*np.sqrt(N)*norm(rho*U,'fro');

    if k == 1 or k % 10 == 0:
        print k, r_norm, eps_pri, s_norm, eps_dual, admm_cost[-1]

    if r_norm < eps_pri and s_norm < eps_dual:
        break

print norm(V, 'fro'), norm(X_1,'fro')
print np.count_nonzero(S), np.count_nonzero(X_2)

print np.linalg.matrix_rank(L), np.linalg.matrix_rank(X_3)

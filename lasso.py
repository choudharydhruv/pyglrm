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
    #permutation gives random numbers in between 0, m*n
    rand_seq = permutation(m*n)[:nnz]
    #calculating row, col through a simple hash
    row  = rand_seq / n
    col  = rand_seq % n
    data = ones(nnz, dtype='int8')
    # duplicate (i,j) entries will be summed together
    # CSR matrix Compressed sparse row matrix

    #csr_matrix stores a sparse representation in a variable data
    #csr_matrix has many attributes like M.data, M.nnz
    #To convert to real matrix you need tocall toArray() or todense()

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

#np.random.seed(seed=58417329)

m=500
n=2500

x0 = sprandn(n,1,0.05).todense();
A = np.random.randn(m,n)
#np.sum calculates sum across all elements unless an axis is provided
col_norm = np.sqrt(np.sum(np.square(A),0))
A =  A/col_norm
v = np.sqrt(0.001)*np.random.randn(m,1)
b = np.dot(A,x0) + v

snr = norm(np.dot(A,x0))**2 / (norm(v)**2)
print "Solving the instance with {} examples {} variables".format(m,n)
print "nnz(x0)={} SNR={}".format(np.count_nonzero(x0), snr)

# np.linalg.norm can accept ord=1,2,...,'nuc', default is Frob norm
gamma_max = norm(np.dot(A.T, b), ord=np.inf)
gamma = 0.1*gamma_max

AtA = np.dot(A.T, A)
Atb = np.dot(A.T, b)

#print AtA, Atb
MAX_ITER = 100
ABSTOL   = 1e-4
RELTOL   = 1e-2

def objective(_A, _b, _gamma, _x, _z):
    return 0.5 * np.sum(np.square(np.dot(_A,_x)-_b)) + _gamma * norm(_z,1);


################## Proximal gradient ##################
def fcost(_A,u,_b):
    return 0.5 * np.sum(np.square(np.dot(_A,u) - _b))

def prox_l1(v,lamb):
    # np.maximum does elementwise maximum
    return np.maximum(0, v - lamb) - np.maximum(0, -v - lamb)

reg = 1 #lambda
beta= 0.5
x = np.zeros((n,1))
xprev = x
prox_optval = []
for k in range(0,MAX_ITER):
    while True:
        grad_x = np.dot(AtA, x) - Atb
        z = prox_l1(x - reg*grad_x, reg*gamma)
        if fcost(A,z,b) <= (fcost(A,x,b) + np.dot(grad_x.T, z-x) + (1/(2*reg)) * np.sum(np.square(z-x))):
            break
        reg = beta*reg
    xprev = x
    x = z

    prox_optval.append(objective(A, b, gamma, x, x))
    if k>1 and np.abs(prox_optval[k] - prox_optval[k-1]) < ABSTOL:
        break

x_prox = x
p_prox = prox_optval[-1]

print np.count_nonzero(x_prox), np.count_nonzero(x0), len(prox_optval), p_prox

################## Fast proximal gradient #####################
reg = 1 #lamb

x = np.zeros((n,1))
xprev = x
prox_optval = []
for k in range(0,MAX_ITER):
    y = x + (1.0*k/(k+3))*(x-xprev)  # multiplying by 1. is very important other wise the whole term is zero
    while True:
        grad_y = np.dot(AtA, y) - Atb
        z = prox_l1(y - reg*grad_y, reg*gamma)
        if fcost(A,z,b) <= (fcost(A,y,b) + np.dot(grad_y.T, z-y) + (1/(2*reg)) * np.sum(np.square(z-y))):
            break
        reg = beta*reg
    xprev = x
    x = z

    prox_optval.append(objective(A, b, gamma, x, x))
    if k>1 and np.abs(prox_optval[k] - prox_optval[k-1]) < ABSTOL:
        break

x_prox = x
p_prox = prox_optval[-1]

print np.count_nonzero(x_prox), np.count_nonzero(x0), len(prox_optval), p_prox

################## ADMM  ####################
def factor(_A, rho):
    _m, _n = _A.shape
    if m >= n:
       L = np.linalg.cholesky(np.dot(A.T,A) + rho*np.eye(n,n))
    else:
       L = np.linalg.cholesky(np.eye(m,m) + 1/rho*(np.dot(A, A.T)))

    L = L
    U = L.T
    return L,U

reg = 1.0 #lambda
rho = 1/reg

x = np.zeros((n,1))
z = np.zeros((n,1))
u = np.zeros((n,1))

L, U = factor(A, rho)

admm_optval = []

'''
We can cache the matrix inversion -> inv(reg*AtA + I)
Alternatively we can cache the rho notation -> inv(AtA +rho*I)

As long as we follow the same notation during x-update

Xk+1 = cached * (reg*Atb + (z-u) )
OR
Xk+1 = cached * (Atb + rho*(z-u) )

These two are equivaent because the lambda and rho cancel out
one factor is an inverse other is not.
'''

cached =  np.linalg.pinv(AtA + rho*np.eye(n,n))  # rho notation
#cached =  np.linalg.pinv(reg*AtA + np.eye(n,n)) # lambda notation

for k in range(0,MAX_ITER):

    # x-update
    ''' This soln is using cholesky factorization
    q = Atb + rho*(z - u)
    if m >= n:
        x = np.linalg.lstsq(U, np.linalg.lstsq(L, q))
    else:
        tmp = np.linalg.lstsq(L, np.dot(A,q))[0]
        x = reg*(q - reg * np.dot(A.T, np.linalg.lstsq(U , tmp)[0]) )
    '''
    x = np.dot( cached , (Atb + rho*(z - u))) #rho notation
    #x = np.dot( cached , (reg*Atb + (z - u))) #lambda notation

    # z-update
    zold = z;
    z = prox_l1(x + u, reg*gamma)

    # u-update
    u = u + x - z

    #diagnostics, reporting, termination checks
    admm_optval.append(objective(A, b, gamma, x, z))
    r_norm   = norm(x - z)
    s_norm   = norm(-rho*(z - zold))
    eps_pri  = np.sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z))
    eps_dual = np.sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if r_norm < eps_pri and s_norm < eps_dual:
         break;

x_admm = z;
p_admm = admm_optval[-1]
print np.count_nonzero(x_admm), np.count_nonzero(x0), len(admm_optval), p_admm

'''
This file implements a bunch of proximal operators
'''


def prox_l1(v,lamb):
    # proximal operator for the L1 norm
    # np.maximum does elementwise maximum
    return np.maximum(0, v - lamb) - np.maximum(0, -v - lamb)


def prox_sum_square(v, lamb):
    # PROX_SUM_SQUARE    Proximal operator of sum-of-squares.

    # prox_sum_square(v,lamb) is the proximal operator of
    # (1/2)||.||_2^2 with parameter lambda.

    return (1.0/(1 + lamb))*v

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

def project_box(v, l, u):
    # PROJECT_BOX    Project a point onto a box (hyper-rectangle).
    # project_box(v,l,u) is the projection of v onto
    # the set { x | l <= x <= u }.

    return np.maximum(l, np.minimum(v, u))

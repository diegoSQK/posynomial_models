import numpy as np
from cvxpy import *

def nnrsqrt_LASSO(phi, y, s, l, e):
    #returns a sparse coefficient vector x
    #model overparameterized with n possible exponent vectors for each measurement
    #parameters: 
        #phi: m by n data matrix with element ij := w(i)^(a_j) where w is the input
        #y: m-length vector of output observations
        #s: sqrt-LASSO regularization scalar
        #l: n-length vector of l1-norm regularization weights
        #e: stopping criterion: solution returned will be e-suboptimal
    print "Solving LASSO problem..."

    m = phi.shape[0]
    n = phi.shape[1]

    #Set up sqrt-LASSO matrices
    phi_t = np.r_[phi, s*np.eye(n)]
    y = np.matrix(y).T
    y_t = np.r_[y, np.matrix(np.zeros(n)).T]
    x = np.zeros(n)

    #sequential coordinate descent
    h = -np.dot(phi_t.T, y_t)
    c = norm(y_t, ord=2)**2
    for _ in range(5000):
    #while True:
        #formulate and check stopping criterion
        #Currently too expensive...?
        """
        u_t = np.subtract(np.matrix(np.dot(phi_t, x)).T, y_t)
        u_t /= norm(u_t, ord=2)
        cond = True
        if np.less(np.ravel(np.dot(phi_t.T, u_t)), -l).any():
            J = []
            for i in range(r):
                j = np.dot(phi_t[:,i], u_t)
                if j < -l[i]:
                    J.append(l[i]/abs(j))
            b = min(J)
        else:
            b = 1
        u = u_t * float(b)
        p = norm(np.dot(phi_t, x) - y_t, ord=2) + np.dot(l, np.abs(x))
        d = -np.dot(y_t.T, u)
        if p - d <= e:
            print "broke"
            break"""

        #perform coordinate descent
        y_norm = norm(y_t, ord=2)
        y_norm_sq = y_norm**2
        for i in range(n):
            z = 0
            phi_i = phi_t[:,i]
            phi_norm_sq = norm(phi_i, 2)**2 
            prod = phi_norm_sq*x[i] - h[i]
            yi_nsq = phi_norm_sq*(x[i]**2) + c - 2*x[i]*h[i]
            if prod > l[i] * np.sqrt(yi_nsq):
                z = (prod/phi_norm_sq) - (l[i]/phi_norm_sq)*np.sqrt((phi_norm_sq*yi_nsq - prod**2)/(phi_norm_sq - l[i]**2))

            delta = z - x[i]
            x[i] = z
            c += phi_norm_sq*(delta**2) + 2*delta*h[i]
            phi_ir = np.reshape(phi_i, (phi_i.shape[0],1))
            h += np.dot(phi_t.T, phi_ir) * delta

        #obj = norm(np.dot(phi_t, x), 2) + np.dot(l, np.abs(x))
        #print obj

    return x
    
def recover_posynomial(w, y, a, l):
    #recovers a posynomial model for data
    #parameters:
        #w: observed input vectors: n by m matrix, m measurements of n parameters
        #y: observed output values
        #a: list of possible exponent vectors obtained by Cartesian product of sets
            #of possibilities for each exponent
    print "Formulating LASSO problem..."
    phi = np.zeros((w.shape[1], len(a)))
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            w_i = np.ravel(w[:,i])
            a_j = a[j]
            p = 1
            for k in range(w_i.size):
                p *= w_i[k]**a_j[k]
            phi[i,j] = p
    s = l/10.0
    return nnrsqrt_LASSO(phi, y, s, l, 1)

def pow_vec(w, a):
    #computes monomial vector^vector operation
    p = 1
    for i in range(len(w)):
        p *= w[i][0,0]**a[i] 
    return p

def f(w):
    #arbitrary posynomial function for generating example data
    a1 = [-1, 1, 0]
    a2 = [1, 2, -2]
    a3 = [1, -1, 2]
    return 12*pow_vec(w, a1) + 5*pow_vec(w, a2) + 1.4*pow_vec(w, a3)

def rand_func(n, d, poss_exp):
    #generate a random posynomial function of n terms in dimension d
    #exponents are randomly selected from poss_exp
    coefficients = np.random.uniform(0, 20, n)
    exponents = []
    for _ in range(n):
        exponents.append(np.random.choice(poss_exp, d))
    
    def f(w):
        s = 0
        for i in range(n):
            s += coefficients[i]*pow_vec(w, exponents[i])
        return s

    return f, coefficients, exponents

def posynomial(coeffs, exps, sample):
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i]*pow_vec(sample, exps[i])
    return y

def cartesian(arrays, out=None):
    #returns Cartesian product of list of n input arrays
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([a.size for a in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n/arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def test():
    #generates data plus Gaussian noise from a posynomial function and attempts
    #to recover the function
    print "Generating data..."
    a1 = np.linspace(-2, 2, 5)
    a2 = np.linspace(-2, 2, 5)
    a3 = np.linspace(-2, 2, 5)

    a = cartesian([a1,a2,a3])

    w = np.matrix(np.random.rand(3)*10).T
    y = np.array(f(w))
    for i in range(10):
        samp = np.matrix(np.random.rand(3)*10).T
        w = np.c_[w, samp]
        out = f(samp)
        y = np.append(y, out)

    return recover_posynomial(w, y, a, 30*np.ones(len(a)))

def main():
    for n in range(1, 6):
        poss_exp = np.linspace(-2, 2, 5)
        f, coeffs, exps = rand_func(n, 3, poss_exp)
        a = cartesian([poss_exp]*3)
    
        w = np.matrix(np.random.rand(3)*10).T
        y = np.array(f(w))
        for i in range(10):
            samp = np.matrix(np.random.rand(3)*10).T
            w = np.c_[w, samp]
            out = f(samp)
            y = np.append(y, out)

        fhat = recover_posynomial(w, y, a, 30*np.ones(len(a)))
        print "original terms: ", n
        print coeffs
        print map(list, exps)
        print "recovered terms: ", np.linalg.norm(fhat, 0)
        print [c for c in fhat if c > 0]
        print [list(a[i]) for i in range(len(fhat)) if fhat[i] > 0]




    




'''Some vector calculus operators defined with Sympy.'''
# Note, that for now vector expression in sympy is () and tensor
# is ((), ()). This is just preliminary - neither of the two can be manipulated
# any further. However they can be handled by dolfincode and turned into
# Expression.

from sympy_utils import is_sympy
from sympy import symbols, diff

X = symbols('x y z')
x, y, z = X


def sDx(f, i):
    'Partial derivative w.r.t to x_i.'
    # Check if sympy and scalar!
    assert is_sympy(f)

    # Allow sDx(f, x)
    return diff(f, X[i]) if isinstance(i, int) else diff(f, i)


def sGrad(f):
    'Gradient of scalar f.'
    assert is_sympy(f)

    # Gradient only for 2d and 3d functions. Returns `vector`, ie tuple
    if z in f.free_symbols:
        d = 3
    else:
        if y not in f.free_symbols:
            assert False
        else:
            d = 2

    grad_f = []
    for x_i in X[:d]:
        grad_f.append(sDx(f, x_i))

    return tuple(grad_f)


def sDiv(f):
    'Divergence of vector f.'

    # Vector is represented as tuple
    assert isinstance(f, tuple)

    # Each component is sympy
    assert all(is_sympy(f_i) for f_i in f)

    # Length of tuple tells if this is 2d vector or 3d and it is assumed
    # that the former is function of x, y but not z.
    d = len(f)
    div_f = sDx(f[0], X[0])
    for f_i, x_i in zip(f[1:], X[1:d]):
        div_f += sDx(f_i, x_i)
    return div_f


def sLaplace(f):
    'Laplacian of scalar f.'
    assert is_sympy(f)

    laplace_f = sDx(sDx(f, X[0]), X[0])         # d^2 f/ d x^2
    for x_i in X[1:]:
        laplace_f += sDx(sDx(f, x_i), x_i)
    return laplace_f


def sRot(f):
    'Rot of 2d vector f. Rot(f) is scalar.'
    assert isinstance(f, tuple) and len(f) == 2
    for fi in f:
        assert is_sympy(f)
        assert z not in f.free_symbols

    rot_f = sDx(f[1], 0) - sDx(f[0], 1)
    return rot_f

def sCurl(f):
    'Curl of 2d scalar of 3d vector. Curl(f) is always vector.'
    # Handle vector
    if isinstance(f, tuple):
        assert len(f) == 3
        assert all(is_sympy(f_i) for f_i in f)

        curl_f = (sDx(f[2], 1) - sDx(f[1], 2),
                  sDx(f[0], 2) - sDx(f[2], 0),
                  sDx(f[1], 0) - sDx(f[0], 1))

        return curl_f

    else:
        assert is_sympy(f)
        assert z not in f.free_symbols

        curl_f = (sDx(f, 1), -sDx(f, 0))

        return curl_f

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    from sympy import sin, cos

    f = sin(x)*cos(y)
    print 'df/dx', sDx(f, 0)
    print 'df/dy', sDx(f, y)
    grad_f = sGrad(f)
    print 'graf(f)', grad_f

    div_f = sDiv(grad_f)
    print 'div(gra_f)', div_f

    laplace_f = sLaplace(f)
    print 'laplace(f)', laplace_f

    curl_f = sCurl(f)
    print 'curl(f)', curl_f

'''Some vector calculus operators defined using weighted averaging.'''

from weight_linalg import *
from utils import *
from dolfin import plot

class CachedwDx(object):
    '''
    This class provides same functionality as wDx but it caches computed
    projection matrices for efficiency.
    '''
    def __init__(self):
        self.cache = [0]*3  # Init empty slots for three matrices
        self.V = None
        self.hit_count = 0

    def __call__(self, f, i, weight_type='harmonic'):
        assert (0 <= i < 3)

        assert is_in_space(f, family='Lagrange', degree=1, shape=())

        V = f.function_space()
        mesh = V.mesh()
        Dx_f = Function(V)

        cache_hit = False
        # I computed some matrix before
        if self.cache[i]:
            # The space of f has not changed from last __call__ and
            # cached matrix is compatible with V
            if self.is_same_space(V) and self.cache[i].size(0) == V.dim():
                self.hit_count += 1
                cache_hit = True

        if not cache_hit:
            # Build new matrix if not cache hit
            WP = build_WP(mesh, weight_type=weight_type, i=i)

            # Cache it and the circumstances
            self.cache[i] = WP
            self.V = V
        else:
            WP = self.cache[i]

        # Get the gradient as vector-matrix product
        WP.mult(f.vector(), Dx_f.vector())
        return Dx_f

    def is_same_space(self, V):
        'Rather naive way of comparing cached and current function spac.'
        dim_match = V.dim() == self.V.dim()
        cell_match = V.mesh().num_cells() == self.V.mesh().num_cells()
        return dim_match and cell_match


def wDx(f, i, weight_type='harmonic'):
    '''
    Given f, a scalar function in CG1(mesh) return its weight-averaged
    Dx(f, i). wDx(f, i) is a function in CG1(mesh).
    '''
    assert is_in_space(f, family='Lagrange', degree=1, shape=())

    V = f.function_space()
    mesh = V.mesh()
    Dx_f = Function(V)

    # Get the gradient as vector-matrix product
    WP = build_WP(mesh, weight_type=weight_type, i=i)
    WP.mult(f.vector(), Dx_f.vector())

    return Dx_f

#########
wDx = CachedwDx()
#########

def wGrad(f, asvector=True, weight_type='harmonic'):
    '''
    Given f, a scalar function in CG1(mesh) return its weight-averaged
    grad(f). wGrad(f) is a function in [CG1(mesh)]^d. Optionally components
    of gradient can be returned as functions in CG1(mesh)
    '''
    assert is_in_space(f, family='Lagrange', degree=1, shape=())

    V = f.function_space()
    mesh = V.mesh()
    gdim = mesh.geometry().dim()
    assert gdim > 1

    # Build components grad(f) = (Dx(f, 0), Dx(f, 1), ...)
    Grad_fi = []
    for i in range(gdim):
        Dx_fi = wDx(f, i, weight_type)
        Grad_fi.append(Dx_fi)

    # Return components or create vector
    if not asvector:
        return Grad_fi
    else:
        # Assign components to vector
        W = VectorFunctionSpace(mesh, 'CG', 1)
        Grad_f = Function(W)
        assigner = FunctionAssigner(W, gdim*[V])
        assigner.assign(Grad_f, Grad_fi)

        return Grad_f


def wDiv(f, weight_type='harmonic'):
    '''
    Given f, a vector function in [CG1(mesh)]^d return its weight-averaged
    div(f). wDiv(f) is a function in CG1(mesh). Alternatively f can be
    components of vector function, each one in CG1(mesh). This case allows
    d = 1.
    '''
    # See if we got list of components
    if isinstance(f, list):
        assert len(f) in [1, 2, 3]
        # Other assertions will fail in Dx

        # Create a function space for divergence
        mesh = f[0].function_space().mesh()
        V = FunctionSpace(mesh, 'CG', 1)
        Div_f = Function(V)

        # it is assumed that f = [f0, f1, f2] so that div(f) = d fi/ d xi
        for i, fi in enumerate(f):
            Div_f.vector()[:] += wDx(fi, i, weight_type).vector()

        return Div_f

    else:
        assert (is_in_space(f, family='Lagrange', degree=1, shape=(2, )) or
                is_in_space(f, family='Lagrange', degree=1, shape=(3, )))

        gdim = f.function_space().mesh().geometry().dim()
        assert gdim > 1

        # Decompose into components and call self
        f_is = list(f.split(True))
        return wDiv(f_is)


def wLaplace(f, weight_type='harmonic'):
    '''
    Given f, a vector function in [CG1(mesh)]^d return its weight-averaged
    div(f). wDiv(f) is a function in CG1(mesh).
    '''
    assert is_in_space(f, family='Lagrange', degree=1, shape=())

    # Compute laplacian from definition
    Laplace_f = wDiv(wGrad(f, weight_type=weight_type, asvector=True),
                     weight_type)

    return Laplace_f


def wCurl(f, asvector=True, weight_type='harmonic'):
    '''
    In 2d, given a scalar function from CG1(mesh), curl(f) = R.grad(f) where
    R = [[0, 1], [-1, 0]]. Thus curl(f) is a function in [CG1(mesh)]^2.

    In 3d, given a vector function from [CG1(mesh)]^3, curl(f)=e_{ijk} d_j f_k.
    This curl(f) is a vector in [CG1(mesh)]^3.
    '''
    is_list = False
    if isinstance(f, list):
        f_vector = len(f) == 3
        f_scalar = False
        is_list = True
    else:
        f_scalar = is_in_space(f, family='Lagrange', degree=1, shape=())
        f_vector = is_in_space(f, family='Lagrange', degree=1, shape=(3, ))

    assert f_scalar or f_vector

    # Vector as list and scalar are handled similarly
    if (is_list and f_vector) or f_scalar:
        if f_scalar:
            # Check geometry
            mesh = f.function_space().mesh()
            gdim = mesh.geometry().dim()
            assert gdim == 2

            # Create components
            Curl_f0 = wDx(f, 1, weight_type)
            Curl_f1 = wDx(f, 0, weight_type)  # -Dx(f, 1)
            Curl_f1.vector()[:] *= -1

            Curl_fi = [Curl_f0, Curl_f1]
        else:
            # Check geometry
            mesh = f[0].function_space().mesh()
            gdim = mesh.geometry().dim()
            assert gdim == 3

            # Create 0th component
            Curl_f0 = wDx(f[2], 1, weight_type)
            Curl_f0.vector().axpy(-1, wDx(f[1], 2, weight_type).vector())
            # Create 1th component
            Curl_f1 = wDx(f[0], 2, weight_type)
            Curl_f1.vector().axpy(-1, wDx(f[2], 0, weight_type).vector())
            # Create 2th component
            Curl_f2 = wDx(f[1], 0, weight_type)
            Curl_f2.vector().axpy(-1, wDx(f[0], 1, weight_type).vector())

            Curl_fi = [Curl_f0, Curl_f1, Curl_f2]

        # Return vector or components
        if not asvector:
            return Curl_fi
        else:
            # Assign components to vector
            d = len(Curl_fi)
            W = VectorFunctionSpace(mesh, 'CG', 1)
            Curl_f = Function(W)
            V = FunctionSpace(mesh, 'CG', 1)
            assigner = FunctionAssigner(W, [V]*d)
            assigner.assign(Curl_f, [Function(V, Curl_fi[i].vector())
                                     for i in range(d)])  # TODO, why?

            return Curl_f

    # f is a vector given by VectorFunction...
    else:
        # Decompose are call self
        f_is = list(f.split(True))
        return wCurl(f_is, weight_type)


def wRot(f, weight_type='harmonic'):
    '''
    For vector function in [CG1(mesh)]^2 return rot(f) = div(R.f)
    where R = [[0, 1], [-1, 0]]. Rot(f) is a scalar function in CG1(mesh).
    '''
    # If f given by components
    if isinstance(f, list):
        assert len(f) == 2

        mesh = f[0].function_space().mesh()
        gdim = mesh.geometry().dim()
        assert gdim == 2

        Rot_f = Dx(f[1], 0, weight_type)
        Rot_f.vector().axpy(-1, Dx(f[0], 1, weight_type).vector())

        return Rot_f
    else:
        assert is_in_space(f, family='Lagrange', degree=1, shape=(2, ))

        mesh = f.function_space().mesh()
        gdim = mesh.geometry().dim()
        assert gdim == 2

        Rot_f = Dx(f[1], 0)
        Rot_f.vector().axpy(-1, Dx(f[0], 1))

        # Decompose into components and call self
        f_is = list(f.split(True))
        return Rot_f(f_is, weight_type)


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    mesh = UnitSquareMesh(40, 40)
    V = FunctionSpace(mesh, 'CG', 1)

    grad_f = wGrad(f)
    plot(grad_f)

    div_f = wDiv(grad_f)
    plot(div_f)

    laplace_f = wLaplace(f)
    plot(laplace_f)

    curl_f = wCurl(f)
    plot(curl_f)

    interactive()

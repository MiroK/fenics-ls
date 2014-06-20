from convergence import test_differential_operators, grads, laplacians
from utils import interior_cells
from sympy_utils import dolfincode
from weight_calculus import *
from sympy_calculus import *
from math import log as ln
from dolfin import *
import sympy as sp


def normal_curvature_convergence(f, mesh, norm, keep_boundary, n=4):
    '''
    Given f - the level set function as sympy expression compute convergence
    rate of weighted normal and weighted curvature in the norm. Keep boundary
    is used to integrate the error away from the boundary or over the entire
    domain.
    '''
    assert is_sympy(f)

    n_rates, n_fig =\
        test_differential_operators(operators=grads,
                                    norm_type=norm,
                                    f=f,
                                    mesh=mesh,
                                    n_refinements=n,
                                    keep_boundary=keep_boundary)

    k_rates, k_fig =\
        test_differential_operators(operators=laplacians,
                                    norm_type=norm,
                                    f=f,
                                    mesh=mesh,
                                    n_refinements=n,
                                    keep_boundary=keep_boundary)

    return n_rates, k_rates

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    x, y = sp.symbols('x y')

    f_sympy = sp.sqrt((x-3)**2 + (y-3)**2) - 1
    f_mesh = UnitSquareMesh(8, 8)
    f_norm = 'L2'
    keep = False

    n, k = normal_curvature_convergence(f=f_sympy,
                                        mesh=f_mesh,
                                        norm=f_norm,
                                        keep_boundary=keep,
                                        n=4)

    print 'Normal convergence', n
    print 'Curvature convergence', k


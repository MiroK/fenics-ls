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

    print 'Normal'
    n_rates, n_fig =\
        test_differential_operators(operators=grads,
                                    norm_type=norm,
                                    f=f,
                                    mesh=mesh,
                                    n_refinements=n,
                                    keep_boundary=keep_boundary)

    print 'Curvature'
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

    print 'Unstructured'
    for f_norm in []:#['L2', 'L1', 'Linf']:
        for keep in [True, False]:
            f_mesh = Mesh('square.xml')

            print 'With bdry', keep, 'Norm', f_norm

            n, k = normal_curvature_convergence(f=f_sympy,
                                                mesh=f_mesh,
                                                norm=f_norm,
                                                keep_boundary=keep,
                                                n=4)
            print '-------------------------------------------------------'

    print 'Structured'
    for diagonal in []: #['left', 'right', 'crossed']:
        for f_norm in ['L2', 'L1', 'Linf']:
            for keep in [True, False]:
                f_mesh = UnitSquareMesh(8, 8, diagonal)

                print 'Diagonal', diagonal, 'With bdry', keep, 'Norm', f_norm

                n, k = normal_curvature_convergence(f=f_sympy,
                                                    mesh=f_mesh,
                                                    norm=f_norm,
                                                    keep_boundary=keep,
                                                    n=4)
                print '-------------------------------------------------------'

    print 'Structured'
    f_sympy = sp.sqrt((x-1.5)**2 + (y-1.5)**2 + (z-1.5)**2) - 1
    for f_norm in []:#['L2', 'L1', 'Linf']:
        for keep in [True, False]:
            f_mesh = UnitCubeMesh(1, 1, 2)

            print 'With bdry', keep, 'Norm', f_norm

            n, k = normal_curvature_convergence(f=f_sympy,
                                                mesh=f_mesh,
                                                norm=f_norm,
                                                keep_boundary=keep,
                                                n=4)
            print '-------------------------------------------------------'


    print 'Unstructured'
    f_sympy = sp.sqrt((x-1.5)**2 + (y-1.5)**2 + (z-1.5)**2) - 1
    for f_norm in []:#['L2', 'L1', 'Linf']:
        for keep in [True, False]:
            f_mesh = Mesh('cube.xml')

            print 'With bdry', keep, 'Norm', f_norm

            n, k = normal_curvature_convergence(f=f_sympy,
                                                mesh=f_mesh,
                                                norm=f_norm,
                                                keep_boundary=keep,
                                                n=4)
            print '-------------------------------------------------------'

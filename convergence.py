'''Convergence-rate test'''

from collections import namedtuple
import matplotlib.pyplot as plt
from sympy_utils import dolfincode
from utils import error_norm
from weight_calculus import *
from sympy_calculus import *
from math import log as ln
from dolfin import *
import numpy as np

set_log_level(WARNING)

DifferentialOperators = namedtuple('DifferentialOperators',
                                   ['domain', 'range',
                                    'sympyD', 'uflD', 'weightD'])


def test_differential_operators(operators,
                                f,
                                mesh,
                                n_refinements,
                                norm_type='L2',
                                keep_boundary=True):
    '''
    Perform convergence test. Operators is a collection with
    sympy/ufl/weighted-differential-operators D and their domain and range.
    Operators D are used respectively to compute exact solution, projected
    solution and weighted solution D(f). Convergence rate is estimated in
    norm_type-rnom on n_refiniments-times refined mesh. Set keep_boundary=False
    to integrate the error only over interior of the domain.
    '''
    assert isinstance(operators, DifferentialOperators)

    # Represent f as Cxpression
    f_expr = Expression(dolfincode(f))

    # Compute exact solution with sympy and represent as Expression
    u = operators.sympyD(f)
    u = Expression(dolfincode(u))

    results = np.zeros((n_refinements, 3+2))  # Table: row = h, ep, ew, rp, rw
    for level in range(n_refinements):
        mesh = refine(mesh)

        # Compute the mesh form error integration
        i_cells=None
        if not keep_boundary:
            i_cells = interior_cells(mesh, width=1)

        # Represent f on CG1
        # D goes from U ---> V
        if operators.domain == 'scalar':
            U = FunctionSpace(mesh, 'CG', 1)
        elif operators.domain == 'vector':
            U = VectorFunctionSpace(mesh, 'CG', 1)
        else:
            assert False

        if operators.range == 'scalar':
            V = FunctionSpace(mesh, 'CG', 1)
        elif operators.range == 'vector':
            V = VectorFunctionSpace(mesh, 'CG', 1)
        else:
            assert False

        # Get f in domain of D = U
        fh = interpolate(f_expr, U)

        # Get the projected solution in range D = V
        u_proj = project(operators.uflD(fh), V)

        # Get the weighted solution
        u_weight = operators.weightD(fh)

        # Get error
        error_proj =\
            error_norm(u, u_proj, norm_type=norm_type, mesh_f=i_cells,
                       subdomain=1)
        error_weight =\
            error_norm(u, u_weight, norm_type=norm_type, mesh_f=i_cells,
                       subdomain=1)

        results[level, 0] = mesh.hmin()
        results[level, 1] = error_proj
        results[level, 2] = error_weight

    # Postprocessing
    # Print convergence rates
    print '  h  \t  proj  \t  weight  '
    for i in range(n_refinements-1):
        h_diff = ln(results[i+1, 0]/results[i, 0])
        rate_proj = ln(results[i+1, 1]/results[i, 1])/h_diff
        rate_weight = ln(results[i+1, 2]/results[i, 2])/h_diff
        results[i+1, 3] = rate_proj
        results[i+1, 4] = rate_weight
        print '%.4E %.2f %.2f' % (results[i+1, 0], rate_proj, rate_weight)

    # Plot convergence rates
    fig = plt.figure()
    axis = fig.gca()

    h = results[:, 0]
    ep = results[:, 1]
    ew = results[:, 2]
    linear = h**1*ep[-1]
    quadratic = h**2*ep[-1]

    axis.loglog(h, ep, 'r*-', label='$project, %.2f$'% results[-1, 3])
    axis.loglog(h, ew, 'b*-', label='$weight, %.2f$' % results[-1, 4])
    axis.loglog(h, linear, 'k--')
    axis.loglog(h, quadratic, 'k--')
    axis.set_xlabel('$h$')
    axis.set_ylabel('$||e||$')
    axis.legend(loc='best')
    plt.title('Convergence rates in %s norm' % norm_type.upper())
    plt.show()

    # Plot the beast
    # plot(u, mesh=mesh)
    # plot(u_proj)
    # plot(u_weight)
    # interactive()

    return results, fig

# Create differential operators for test
laplace = lambda u: div(grad(u))

laplacians =\
    DifferentialOperators(domain='scalar', range='scalar',
                          sympyD=sLaplace, uflD=laplace, weightD=wLaplace)

d_dxis =\
    lambda i: DifferentialOperators(domain='scalar', range='scalar',
                                    sympyD=lambda f: sDx(f, i),
                                    uflD=lambda f: Dx(f, i),
                                    weightD=lambda f: wDx(f, i))
grads = \
    DifferentialOperators(domain='scalar', range='vector',
                          sympyD=sGrad, uflD=grad, weightD=wGrad)

divs = \
    DifferentialOperators(domain='vector', range='scalar',
                          sympyD=sDiv, uflD=div, weightD=wDiv)

curls = \
    DifferentialOperators(domain='vector', range='vector',
                          sympyD=sCurl, uflD=curl, weightD=wCurl)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Let's test 2d convergence
    from sympy import sin, cos, pi, sqrt

    # Test 2D
    n = 4
    mesh = UnitSquareMesh(8, 8)

    x, y, z = symbols('x y z')
    f_scalar = sin(pi*x)*cos(4*pi*y)**2
    f_vector = (sin(pi*x**2), x**2*sin((x**2 + y)))

    test_differential_operators(operators=d_dxis(0), norm_type='L2',
                                f=f_scalar, mesh=mesh, n_refinements=n,
                                keep_boundary=False)

    # Test curl in 3D
    # mesh = UnitCubeMesh(1, 1, 1)
    # f_vector = (sin(pi*x)*sin(pi*y)*sin(pi*z),
    #             cos(pi*x)*sin(pi*y)*sin(pi*z),
    #            cos(pi*x)*cos(pi*y)*sin(pi*z))

    # test_differential_operators(operators=curls,
    #                             f=f_vector, mesh=mesh, n_refinements=n)

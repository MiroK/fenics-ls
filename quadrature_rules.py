'''Computer proof of logic behind weight matrix.'''

from dolfin import *


def basis_function_integrals(dim, degree):
    'Calculate integrals of basis function of CG_degree over reference cell'

    assert 1 <= dim < 4, 'Only 1d, 2d, 3d!'
    if dim == 1:
        mesh = UnitIntervalMesh(1)
    elif dim == 2:
        mesh = UnitTriangleMesh()
    elif dim == 3:
        mesh = UnitTetrahedronMesh()

    V = FunctionSpace(mesh, 'CG', degree)
    v = TestFunction(V)

    integrals = assemble(v*dx).array()

    # For any dim, if degree == 1 than all the functions' integrals are
    # cell_volume/(dim + 1)
    proposed = [cell.volume()/(dim + 1) for cell in cells(mesh)]
    # In general, functions in higher order spaces have different values of
    # integral and thus much approach to weight matrix is hard to generalize
    return all((near(abs(p-i), DOLFIN_EPS)
                for p, i in zip(proposed, integrals)))

# -----------------------------------------------------------------------------

for dim in [1, 2, 3]:
    print dim, basis_function_integrals(dim=dim, degree=1)

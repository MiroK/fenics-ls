'''Linear operators for weighted grad, div, ...'''

from petsc4py import PETSc
from dolfin import *
import numpy as np

parameters['linear_algebra_backend'] = 'PETSc'


def build_W(mesh, weight_type):
    '''
    Build weighting matrix W for weighted averaging of quantites between
    spaces DG0(mesh) and CG1(mesh).
    '''
    d = mesh.geometry().dim()

    CG1 = FunctionSpace(mesh, 'CG', 1)
    DG0 = FunctionSpace(mesh, 'DG', 0)

    v = TestFunction(CG1)
    u = TrialFunction(DG0)

    alpha = Constant(d+1)        # 1./{basis function value at barycenter}
    # Weak form, alpha makes the integral indep of CG1 basis function value
    if weight_type == 'arithmetic':
        a = alpha*inner(v, u)*dx       # This is just |K|, cell volume, now

    elif weight_type == 'harmonic':
        cell_volume = CellVolume(mesh)
        a = alpha*inner(v, u)/cell_volume**2*dx   # This is 1/|K|
    else:
        raise ValueError('Unsupported weight type {0}'.format(weight_type))

    W = PETScMatrix()
    assemble(a, tensor=W)

    # Now imagine wanting to change the matrix entries so that each value
    # is divided by the row sum of the values, ie, the new row sum is 1
    # and the averaging is consistent
    first_row, last_row = W.local_range(0)   # Global index of row

    # Create new entries
    new_cols = []
    new_vals = []
    for row in range(first_row, last_row):
        cols, vals = W.getrow(row)
        row_sum = np.sum(vals)
        vals /= row_sum
        new_cols.append(np.array(cols, dtype='uintp'))
        new_vals.append(vals)

    # Set new entries
    for row in reversed(range(first_row, last_row)):
        cols = new_cols.pop()
        vals = new_vals.pop()
        W.setrow(row, cols, vals)
    # Without pop there's almost no speed-up
    #for (i, row) in enumerate(range(first_row, last_row)):
    #    cols = new_cols[i]
    #    vals = new_vals[i]
    #    W.setrow(row, cols, vals)

    W.apply('insert')

    return W


def build_P(mesh, i):
    '''
    Build matrix P for computing partial deriveative in i direction of function
    f in CG1(mesh). With F the expansion coefficients of f and G the expansion
    coefficients of partial derivative in DG0, G is computed as G = PF.
    Note that P includes M inverse, M being the mass matrix of DG0.
    '''
    assert i < mesh.geometry().dim()

    DG0 = FunctionSpace(mesh, 'DG', 0)
    CG1 = FunctionSpace(mesh, 'CG', 1)

    cell_volume = CellVolume(mesh)
    u = TestFunction(DG0)
    v = TrialFunction(CG1)

    # Weak form, include cell_volume as M inverse
    a = inner(Dx(v, i), u)/cell_volume*dx

    P = PETScMatrix()
    assemble(a, tensor=P)

    return P


def build_WP(mesh, weight_type, i):
    '''
    Build matrix WP which applied to vector of expansion coefficients of
    CG1(mesh) function f returns vector of expansion coefficients for function
    representing weigh-averaged partial derivative of f.
    '''
    W = build_W(mesh, weight_type)
    P = build_P(mesh, i)

    # Matrix multiply outside by PETSc
    WP_petsc = PETSc.Mat()
    W.mat().matMult(P.mat(), WP_petsc)

    # Come back to PETScMatrix
    WP = PETScMatrix(WP_petsc)

    return WP

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    mesh = UnitSquareMesh(40, 40)

    V = FunctionSpace(mesh, 'CG', 1)
    f = interpolate(Expression('sin(2*pi*x[0])*cos(3*pi*x[1])'), V)
    grad_f = Function(V)

    # An advantage of pure matrix/operator approach if that these can be
    # stored and reused.
    WP = build_WP(mesh, weight_type='harmonic', i=1)
    WP.mult(f.vector(), grad_f.vector())

    plot(f)
    plot(grad_f)
    interactive()

    from fenicstools import weighted_gradient_matrix
    mesh = UnitCubeMesh(2, 2, 2)

    WP = build_WP(mesh, weight_type='harmonic', i=1)
    FWP = weighted_gradient_matrix(mesh, 1)
    print WP.array() - FWP.array()

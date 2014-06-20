'''Common useful functions.'''

from dolfin import CellFunction, FacetFunction, DomainBoundary,\
    SubsetIterator, FunctionSpace, VectorFunctionSpace, Measure, Function,\
    Constant, interpolate, sqrt, cells, assemble, inner, dx

import numpy as np


def is_in_space(f, family, degree, shape):
    '''
    Check f belong to finite element space with elements from
    family, polynomials of degree and shape
    '''
    f_element = f.function_space().ufl_element()

    is_family = f_element.family() == family
    is_degree = f_element.degree() == degree
    is_shape = f_element.value_shape() == shape

    return is_family and is_degree and is_shape


def interior_cells(mesh, width=0):
    '''
    Return cell function that has cells with some facet on boundary marked
    as 0. Other cells are true. Width parameter sets number of sweaps
    where cell that neighbors `0` cell becomes `0`.
    '''
    # Get cell facet connectivity
    tdim = mesh.topology().dim()
    mesh.init(tdim, tdim-1)

    # Mark facets on the boundary
    bdry_facets = FacetFunction('bool', mesh, False)
    BDRY_FACETS = bdry_facets.array()
    DomainBoundary().mark(bdry_facets, True)

    mesh_cells = CellFunction('size_t', mesh, 1)
    is_bdry_cell = lambda cell: any(BDRY_FACETS[cell.entities(tdim-1)])
    for cell in filter(is_bdry_cell, cells(mesh)):
        mesh_cells[cell] = 0

    # Mark neighbors of false cells
    if width:
        mesh.init(tdim, tdim)
        MESH_CELLS = mesh_cells.array()
        for i in range(width):
            # Note, Subsetiterator needs 'size_t' not bool
            false_cells = SubsetIterator(mesh_cells, 0)
            # TODO, filter unique?
            new_false_cells = np.concatenate([cell.entities(tdim)
                                              for cell in false_cells])
            MESH_CELLS[new_false_cells] = 0

    return mesh_cells


def error_norm(u, uh, norm_type, mesh_f=None, subdomain=None):
    degree = uh.ufl_element().degree()
    rank = uh.value_rank()

    if mesh_f is None:
        mesh = uh.function_space().mesh()
    else:
        assert subdomain is not None
        mesh = mesh_f.mesh()

    if rank == 0:
        V = FunctionSpace(mesh, 'DG', degree+3)
    elif rank == 1:
        V = VectorFunctionSpace(mesh, 'CG', degree+3)
    else:
        assert False

    U = interpolate(u, V)
    uh = interpolate(uh, V)
    e = Function(V, U.vector())
    e.vector().axpy(-1, uh.vector())

    dX = Measure('dx')
    volume = assemble(Constant(1)*dX(mesh))

    if mesh_f is not None:
        volume_ = assemble(Constant(1)*dX(subdomain, domain=mesh,
                                       subdomain_data=mesh_f))
        # Only use the restricted mesh if it actually has content
        volume = volume_ if volume_ > 0 else volume
        if volume_ > 0:
            dX = Measure('dx')[mesh_f]
            dX = dX(subdomain)

    norm_type = norm_type.lower()
    if norm_type == 'l1':
        if rank == 0:
            error = assemble(abs(e)*dX)/volume
        else:
            # Well...
            error = sum(assemble(abs(e[i])*dX)/volume
                     for i in range(mesh.geometry().dim()))
    elif norm_type == 'l2':
        error = sqrt(assemble(inner(e, e)*dX))/volume
    elif norm_type == 'linf':
        e.vector().abs()
        error = e.vector().max()
    else:
        assert False

    return error

if __name__ == '__main__':
    from dolfin import plot, UnitSquareMesh

    # Show off interior_cells
    mesh = UnitSquareMesh(20, 20)
    plot(interior_cells(mesh, width=1), interactive=True)

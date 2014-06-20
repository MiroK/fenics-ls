'''Useful SymPy tools.'''

from sympy.printing.ccode import CCodePrinter
from sympy import symbols


def is_sympy(args):
    'Check if args are sympy objects.'
    try:
        return args.__module__.split('.')[0] == 'sympy'
    except AttributeError:
        return False


class DolfinCodePrinter(CCodePrinter):
    '''
    This class provides functionality for converting sympy expression to
    DOLFIN expression. Core of work is done by dolfincode.
    '''
    def __init__(self, settings={}):
        CCodePrinter.__init__(self)

    def _print_Pi(self, expr):
        return 'pi'


def dolfincode(expr, assign_to=None, **settings):
    '''
    Convert sympy expr to string that can be passed to Expression. Thus
    DOLFIN Expression can be created from sympy.
    '''
    # Handle scalars
    if is_sympy(expr):
        dolfin_xs = symbols('x[0] x[1] x[2]')
        xs = symbols('x y z')

        for x, dolfin_x in zip(xs, dolfin_xs):
            expr = expr.subs(x, dolfin_x)

        return DolfinCodePrinter(settings).doprint(expr, assign_to)

    # Recurse if vector or tensor
    elif type(expr) is tuple:
        return tuple(dolfincode(e, assign_to, **settings) for e in expr)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Show typical usecase for sympy to Expression conversion
    from dolfin import Expression, UnitSquareMesh, interactive, plot
    from sympy import sin, cos, exp, sqrt, pi

    x, y, z = symbols('x y z')
    mesh = UnitSquareMesh(10, 10)

    # Sympy scalar
    u = sin(pi*x)*cos(exp(sqrt(x**2) + y**2))*abs(x**y)
    u = dolfincode(u)
    print u  # String for expression
    u = Expression(u)
    plot(u, mesh=mesh, title='u')

    # Sympy `vector`
    v = (sin(pi*x), cos(pi*y))
    v = dolfincode(v)
    print v
    v = Expression(v)
    plot(v, mesh=mesh, title='v')

    # Sympy tensor
    w = ((x, sin(pi*x)), (y, cos(pi*y)))
    w = dolfincode(w)
    print w
    w = Expression(w)
    plot(w[0, :], mesh=mesh, title='w[0, :]')

    interactive()

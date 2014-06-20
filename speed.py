from fenicstools import weighted_gradient_matrix as f_build_WP
from weight_linalg import build_WP
from weight_calculus import wDx, CachedwDx
from dolfin import *


def matrix_build(n):
    'Compare matrix building speed of fenicstools and matrix implementation.'
    mesh = UnitCubeMesh(25, 25, 25)
    timer0 = Timer('Fenicstools')
    timer0.start()
    for i in range(n):
        FWP = f_build_WP(mesh, 0)
    timer0.stop()

    timer1 = Timer('Matrix')
    timer1.start()
    for i in range(n):
        WP = build_WP(mesh, 'harmonic', 0)
    timer1.stop()

    print 'Fenicstools', timing('Fenicstools')/n
    print 'Matrix', timing('Matrix')/n

def function_build(n):
    'Measure Dx compoutation speed. (Re)use matrix/vector.'
    mesh = UnitCubeMesh(25, 25, 25)
    V = FunctionSpace(mesh, 'CG', 1)
    f = interpolate(Expression('sin(x[0])*cos(x[1])*x[2]'), V)
    print V.dim()

    timer0 = Timer('Naive')
    timer0.start()
    for i in range(n):
        Dx_f = wDx(f, 0)
    print 'Naive', timer0.stop()/n

    # Get the gradient as vector-matrix product
    timer1 = Timer('Smart')
    timer1.start()
    WP = build_WP(mesh, weight_type='harmonic', i=0)
    for i in range(n):
        Dx_f = Function(V)
        WP.mult(f.vector(), Dx_f.vector())
    print 'Smart', timer1.stop()/n

    # Get the gradient as vector-matrix product
    timer2 = Timer('Smarter')
    timer2.start()
    WP = build_WP(mesh, weight_type='harmonic', i=0)
    Dx_f = Function(V)
    for i in range(n):
        WP.mult(f.vector(), Dx_f.vector())
    print 'Smarter', timer2.stop()/n

    # Get the gradient as vector-matrix product
    timer3 = Timer('Cached')
    timer3.start()
    cached_Dx = CachedwDx()
    for i in range(n):
        Dx_f = cached_Dx(f, 0)
    print 'Cached', timer3.stop()/n

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # matrix_build(5)
    # Matrix is very much comparable with fenicstools despite
    # being pure python

    function_build(20)
    # 25x25x25 -->
    # 17576
    # Naive 0.791988603486
    # Smart 0.0399125546799
    # Smarter 0.0404151466209
    # Cached 0.0451131960843    # << way to go, so now cached is used evrywhere

    # 50x50x50, single and also on 3 procs
    # 132651
    # Naive 6.06383185542
    # Smart 0.309928147082
    # Smarter 0.313423802238

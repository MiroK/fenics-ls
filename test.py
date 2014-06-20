'''Compare against fenicstools'''

from dolfin import UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh, pi,\
    has_cgal, FunctionSpace, Expression, interpolate
from fenicstools import weighted_gradient_matrix as f_build_WP
from weight_linalg import build_WP
from weight_calculus import CachedwDx, wDx
from types import FunctionType
import numpy as np


def test_structured(tol=1E-15):
    'Test with structured non-uniform mesh. Fenicstools vs matrix.'
    passed = []
    for d in [1, 2, 3]:
        if d == 1:
            mesh = UnitIntervalMesh(20)
            mesh.coordinates()[:] = np.cos(2*pi*mesh.coordinates())
        elif d == 2:
            mesh = UnitSquareMesh(20, 20)
            mesh.coordinates()[:] = np.cos(2*pi*mesh.coordinates())
        else:
            mesh = UnitCubeMesh(5, 5, 5)
            mesh.coordinates()[:, 0] = mesh.coordinates()[:, 0]**3
            mesh.coordinates()[:, 1] = mesh.coordinates()[:, 1]**2
            mesh.coordinates()[:, 2] = mesh.coordinates()[:, 2]**2

        for j in range(d):
            WP = build_WP(mesh, 'harmonic', i=j).array()
            F_WP = f_build_WP(mesh, i=j).array()
            e = np.linalg.norm(WP - F_WP)/WP.shape[0]
            passed.append(e < tol)
    assert(all(passed))

def test_unstructered(tol=1E-15):
    'Test with unstructured mesh. Fenicstools vs matrix.'
    # Test only with CGAL
    if has_cgal():
        # Some additional imports
        from dolfin import CircleMesh, SphereMesh, Point
        passed = []
        for d in [2, 3]:
            if d == 2:
                mesh = CircleMesh(Point(0., 0), 1, 0.5)
            else:
                mesh = SphereMesh(Point(0., 0., 0.), 1, 0.75)

            for j in range(d):
                WP = build_WP(mesh, 'harmonic', i=j).array()
                F_WP = f_build_WP(mesh, i=j).array()
                e = np.linalg.norm(WP - F_WP)/WP.shape[0]
                passed.append(e < tol)
        assert(all(passed))
    else:
        assert True

def test_cached_Dx():
    'Test if cached Dx produces same results as not cached + caches correctly.'
    # If wDx is CachedwDx object and not function, this test should auto-pass
    if not isinstance(wDx, FunctionType):
        assert True
    else:
        # Make comparison
        cached_wDx = CachedwDx()

        mesh = UnitSquareMesh(50, 50)
        V = FunctionSpace(mesh, 'CG', 1)
        f = interpolate(Expression('sin(x[0])*cos(x[1])'), V)

        x = cached_wDx(f, 0)
        y = wDx(f, 0)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 0)  # 1
        y = wDx(f, 0)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 1)
        y = wDx(f, 1)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 0)  # 2
        y = wDx(f, 0)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 1)  # 3
        y = wDx(f, 1)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 1)  # 4
        y = wDx(f, 1)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16


        mesh = UnitSquareMesh(10, 50)
        V = FunctionSpace(mesh, 'CG', 1)
        f = interpolate(Expression('sin(x[0])*cos(x[1])'), V)

        x = cached_wDx(f, 1)
        y = wDx(f, 1)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 0)
        y = wDx(f, 0)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 0)  # 5
        y = wDx(f, 0)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 1)  # 6
        y = wDx(f, 1)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        mesh = UnitCubeMesh(10, 10, 10)
        V = FunctionSpace(mesh, 'CG', 1)
        f = interpolate(Expression('sin(x[0])*cos(x[1])*x[2]'), V)

        x = cached_wDx(f, 0)
        y = wDx(f, 0)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 1)
        y = wDx(f, 1)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 2)
        y = wDx(f, 2)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 2)  # 7
        y = wDx(f, 2)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 2)  # 8
        y = wDx(f, 2)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        x = cached_wDx(f, 0)  # 9
        y = wDx(f, 0)
        assert np.linalg.norm(x.vector().array() - y.vector().array()) < 1E-16

        assert cached_wDx.hit_count == 9

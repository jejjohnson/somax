from operator import (
    add,
    sub,
)

from finitevolx._src.domain.domain import Domain
from finitevolx._src.operators.functional.stagger import (
    batch_domain_limits_transform,
    domain_limits_transform,
    stagger_domain,
)

Nx, Ny = 201, 104
dx, dy = 100, 20
Lx, Ly = (Nx - 1) * dx, (Ny - 1) * dy
xmin, ymin = 0.0, 0.0

H_DOMAIN = Domain(
    xmin=(xmin, ymin), xmax=(Lx, Ly), Lx=(Lx, Ly), Nx=(Nx, Ny), dx=(dx, dy)
)


def test_limits_transform_null():
    xmin, ymin = H_DOMAIN.xmin
    xmax, ymax = H_DOMAIN.xmax
    Lx, Ly = H_DOMAIN.Lx
    Nx, Ny = H_DOMAIN.Nx

    # Case I: None | no stagger
    direction = None
    stagger = False
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx == Nx_
    assert Lx == Lx_
    assert xmin == xmin_
    assert xmax == xmax_

    # Case II: None | stagger
    direction = None
    stagger = True
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx == Nx_
    assert Lx == Lx_
    assert xmin == xmin_
    assert xmax == xmax_


def test_limits_transform_no_stagger():
    xmin, ymin = H_DOMAIN.xmin
    xmax, ymax = H_DOMAIN.xmax
    Lx, Ly = H_DOMAIN.Lx
    Nx, Ny = H_DOMAIN.Nx

    # Case I: right | no stagger
    direction = "right"
    stagger = False
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx == Nx_
    assert Lx == Lx_
    assert xmin + dx == xmin_
    assert xmax + dx == xmax_

    # Case II: left | no stagger
    direction = "left"
    stagger = False
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx == Nx_
    assert Lx == Lx_
    assert xmin - dx == xmin_
    assert xmax - dx == xmax_

    # Case III: inner | no stagger
    direction = "inner"
    stagger = False
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx_ == Nx - 2
    assert Lx_ == Lx - 2 * dx
    assert xmin + dx == xmin_
    assert xmax - dx == xmax_

    # Case IV: outer | no stagger
    direction = "outer"
    stagger = False
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx_ == Nx + 2
    assert Lx_ == Lx + 2 * dx
    assert xmin - dx == xmin_
    assert xmax + dx == xmax_


def test_limits_transform_stagger():
    xmin, ymin = H_DOMAIN.xmin
    xmax, ymax = H_DOMAIN.xmax
    Lx, Ly = H_DOMAIN.Lx
    Nx, Ny = H_DOMAIN.Nx

    # Case I: right | no stagger
    direction = "right"
    stagger = True
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx_ == Nx
    assert Lx_ == Lx
    assert xmin_ == xmin + 0.5 * dx
    assert xmax_ == xmax + 0.5 * dx

    # Case II: left | no stagger
    direction = "left"
    stagger = True
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx_ == Nx
    assert Lx_ == Lx
    assert xmin_ == xmin - 0.5 * dx
    assert xmax_ == xmax - 0.5 * dx

    # Case III: inner | no stagger
    direction = "inner"
    stagger = True
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx_ == Nx - 1
    assert Lx_ == Lx - dx
    assert xmin_ == xmin + 0.5 * dx
    assert xmax_ == xmax - 0.5 * dx

    # Case IV: outer | no stagger
    direction = "outer"
    stagger = True
    xmin_, xmax_, Nx_, Lx_ = domain_limits_transform(
        xmin=xmin, xmax=xmax, dx=dx, Lx=Lx, Nx=Nx, direction=direction, stagger=stagger
    )

    assert Nx_ == Nx + 1
    assert Lx_ == Lx + dx
    assert xmin_ == xmin - 0.5 * dx
    assert xmax_ == xmax + 0.5 * dx


def test_batch_limits_transform_nnull():
    # Case I: None | no stagger
    direction = (None, None)
    stagger = (False, False)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )

    assert Nx_ == H_DOMAIN.Nx
    assert Lx_ == H_DOMAIN.Lx
    assert xmin_ == H_DOMAIN.xmin
    assert xmax_ == H_DOMAIN.xmax

    # Case II: None | stagger
    direction = (None, None)
    stagger = (True, True)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )

    assert Nx_ == H_DOMAIN.Nx
    assert Lx_ == H_DOMAIN.Lx
    assert xmin_ == H_DOMAIN.xmin
    assert xmax_ == H_DOMAIN.xmax


def test_batch_limits_transform_no_stagger():
    # Case I: right | no stagger
    direction = ("right", "right")
    stagger = (False, False)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )

    assert Nx_ == H_DOMAIN.Nx
    assert Lx_ == H_DOMAIN.Lx
    assert xmin_ == tuple(map(add, H_DOMAIN.xmin, H_DOMAIN.dx))
    assert xmax_ == tuple(map(add, H_DOMAIN.xmax, H_DOMAIN.dx))

    # Case II: left | no stagger
    direction = ("left", "left")
    stagger = (False, False)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )

    assert Nx_ == H_DOMAIN.Nx
    assert Lx_ == H_DOMAIN.Lx
    assert xmin_ == tuple(map(sub, H_DOMAIN.xmin, H_DOMAIN.dx))
    assert xmax_ == tuple(map(sub, H_DOMAIN.xmax, H_DOMAIN.dx))

    # Case III: inner | no stagger
    direction = ("inner", "inner")
    stagger = (False, False)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )

    assert Nx_ == tuple(map(lambda x: x - 2, H_DOMAIN.Nx))
    assert Lx_ == tuple(
        map(lambda x: x[0] - 2 * x[1], list(zip(H_DOMAIN.Lx, H_DOMAIN.dx)))
    )
    assert xmin_ == tuple(map(add, H_DOMAIN.xmin, H_DOMAIN.dx))
    assert xmax_ == tuple(map(sub, H_DOMAIN.xmax, H_DOMAIN.dx))

    # Case IV: outer | no stagger
    direction = ("outer", "outer")
    stagger = (False, False)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )

    assert Nx_ == tuple(map(lambda x: x + 2, H_DOMAIN.Nx))
    assert Lx_ == tuple(
        map(lambda x: x[0] + 2 * x[1], list(zip(H_DOMAIN.Lx, H_DOMAIN.dx)))
    )
    assert xmin_ == tuple(map(sub, H_DOMAIN.xmin, H_DOMAIN.dx))
    assert xmax_ == tuple(map(add, H_DOMAIN.xmax, H_DOMAIN.dx))


def test_batch_limits_transform_stagger():
    # Case I: right | no stagger
    direction = ("right", "right")
    stagger = (True, True)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )

    assert Nx_ == H_DOMAIN.Nx
    assert Lx_ == H_DOMAIN.Lx
    assert xmin_ == tuple(
        map(lambda x: x[0] + 0.5 * x[1], list(zip(H_DOMAIN.xmin, H_DOMAIN.dx)))
    )
    assert xmax_ == tuple(
        map(lambda x: x[0] + 0.5 * x[1], list(zip(H_DOMAIN.xmax, H_DOMAIN.dx)))
    )

    # Case II: left | no stagger
    direction = ("left", "left")
    stagger = (True, True)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )
    assert Nx_ == H_DOMAIN.Nx
    assert Lx_ == H_DOMAIN.Lx
    assert xmin_ == tuple(
        map(lambda x: x[0] - 0.5 * x[1], list(zip(H_DOMAIN.xmin, H_DOMAIN.dx)))
    )
    assert xmax_ == tuple(
        map(lambda x: x[0] - 0.5 * x[1], list(zip(H_DOMAIN.xmax, H_DOMAIN.dx)))
    )

    # Case III: inner | no stagger
    direction = ("inner", "inner")
    stagger = (True, True)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )
    assert Nx_ == tuple(map(lambda x: x - 1, H_DOMAIN.Nx))
    assert Lx_ == tuple(map(lambda x: x[0] - x[1], list(zip(H_DOMAIN.Lx, H_DOMAIN.dx))))
    assert xmin_ == tuple(
        map(lambda x: x[0] + 0.5 * x[1], list(zip(H_DOMAIN.xmin, H_DOMAIN.dx)))
    )
    assert xmax_ == tuple(
        map(lambda x: x[0] - 0.5 * x[1], list(zip(H_DOMAIN.xmax, H_DOMAIN.dx)))
    )

    # Case IV: outer | no stagger
    direction = ("outer", "outer")
    stagger = (True, True)
    xmin_, xmax_, Nx_, Lx_ = batch_domain_limits_transform(
        xmin=H_DOMAIN.xmin,
        xmax=H_DOMAIN.xmax,
        dx=H_DOMAIN.dx,
        Lx=H_DOMAIN.Lx,
        Nx=H_DOMAIN.Nx,
        direction=direction,
        stagger=stagger,
    )
    assert Nx_ == tuple(map(lambda x: x + 1, H_DOMAIN.Nx))
    assert Lx_ == tuple(map(lambda x: x[0] + x[1], list(zip(H_DOMAIN.Lx, H_DOMAIN.dx))))
    assert xmin_ == tuple(
        map(lambda x: x[0] - 0.5 * x[1], list(zip(H_DOMAIN.xmin, H_DOMAIN.dx)))
    )
    assert xmax_ == tuple(
        map(lambda x: x[0] + 0.5 * x[1], list(zip(H_DOMAIN.xmax, H_DOMAIN.dx)))
    )


def test_stagger_domain():
    # Case I: right | no stagger
    direction = ("right", "right")
    stagger = (True, True)
    u_domain = stagger_domain(domain=H_DOMAIN, direction=direction, stagger=stagger)

    assert u_domain.Nx == H_DOMAIN.Nx
    assert u_domain.Lx == H_DOMAIN.Lx
    assert u_domain.xmin == tuple(
        map(lambda x: x[0] + 0.5 * x[1], list(zip(H_DOMAIN.xmin, H_DOMAIN.dx)))
    )
    assert u_domain.xmax == tuple(
        map(lambda x: x[0] + 0.5 * x[1], list(zip(H_DOMAIN.xmax, H_DOMAIN.dx)))
    )

    # Case II: left | no stagger
    direction = ("left", "left")
    stagger = (True, True)
    u_domain = stagger_domain(domain=H_DOMAIN, direction=direction, stagger=stagger)

    assert u_domain.Nx == H_DOMAIN.Nx
    assert u_domain.Lx == H_DOMAIN.Lx
    assert u_domain.xmin == tuple(
        map(lambda x: x[0] - 0.5 * x[1], list(zip(H_DOMAIN.xmin, H_DOMAIN.dx)))
    )
    assert u_domain.xmax == tuple(
        map(lambda x: x[0] - 0.5 * x[1], list(zip(H_DOMAIN.xmax, H_DOMAIN.dx)))
    )

    # Case III: inner | no stagger
    direction = ("inner", "inner")
    stagger = (True, True)
    u_domain = stagger_domain(domain=H_DOMAIN, direction=direction, stagger=stagger)

    assert u_domain.Nx == tuple(map(lambda x: x - 1, H_DOMAIN.Nx))
    assert u_domain.Lx == tuple(
        map(lambda x: x[0] - x[1], list(zip(H_DOMAIN.Lx, H_DOMAIN.dx)))
    )
    assert u_domain.xmin == tuple(
        map(lambda x: x[0] + 0.5 * x[1], list(zip(H_DOMAIN.xmin, H_DOMAIN.dx)))
    )
    assert u_domain.xmax == tuple(
        map(lambda x: x[0] - 0.5 * x[1], list(zip(H_DOMAIN.xmax, H_DOMAIN.dx)))
    )

    # Case IV: outer | no stagger
    direction = ("outer", "outer")
    stagger = (True, True)
    u_domain = stagger_domain(domain=H_DOMAIN, direction=direction, stagger=stagger)

    assert u_domain.Nx == tuple(map(lambda x: x + 1, H_DOMAIN.Nx))
    assert u_domain.Lx == tuple(
        map(lambda x: x[0] + x[1], list(zip(H_DOMAIN.Lx, H_DOMAIN.dx)))
    )
    assert u_domain.xmin == tuple(
        map(lambda x: x[0] - 0.5 * x[1], list(zip(H_DOMAIN.xmin, H_DOMAIN.dx)))
    )
    assert u_domain.xmax == tuple(
        map(lambda x: x[0] + 0.5 * x[1], list(zip(H_DOMAIN.xmax, H_DOMAIN.dx)))
    )

    # Case V: None | no stagger
    direction = (None, None)
    stagger = (False, False)
    u_domain = stagger_domain(domain=H_DOMAIN, direction=direction, stagger=stagger)

    assert u_domain.Nx == H_DOMAIN.Nx
    assert u_domain.Lx == H_DOMAIN.Lx
    assert u_domain.xmin == H_DOMAIN.xmin
    assert u_domain.xmax == H_DOMAIN.xmax

    # Case VI: None | stagger
    direction = (None, None)
    stagger = (True, True)
    u_domain = stagger_domain(domain=H_DOMAIN, direction=direction, stagger=stagger)

    assert u_domain.Nx == H_DOMAIN.Nx
    assert u_domain.Lx == H_DOMAIN.Lx
    assert u_domain.xmin == H_DOMAIN.xmin
    assert u_domain.xmax == H_DOMAIN.xmax

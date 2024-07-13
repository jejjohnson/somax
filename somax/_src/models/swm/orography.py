import jax.numpy as jnp


def init_flat_orography(domain):
    return jnp.zeros(domain.shape)


def init_slope(domain):
    x = domain.coords_x
    X = domain.grid[..., 0]
    H = 9000.*2.*jnp.abs((jnp.mean(x)-X)/jnp.max(x))
    return H

def init_gaussian_mountain(domain):
    dx, dy = domain.resolution
    x = domain.coords_x
    y = domain.coords_y
    X, Y = domain.grid[..., 0], domain.grid[..., 1]

    std_mountain_x = 6. * dx # Std. dev. of mountain in x direction (m)
    std_mountain_y = 6. * dy # Std. dev. of mountain in y direction (m)
    H = 5000. * jnp.exp(
        - 0.5 * ((X-jnp.mean(x))/std_mountain_x)**2.
        - 0.5 * ((Y-jnp.mean(y))/std_mountain_y)**2.
    )
    return H


def init_sea_mountain(domain):
    x = domain.coords_x
    y = domain.coords_y
    X, Y = domain.grid[..., 0], domain.grid[..., 1]
    dy = domain.resolution[1]
    std_mountain = 40.0*dy
    H = 9250.*jnp.exp(
        -((X-jnp.mean(x))**2.+(Y-0.5*jnp.mean(y))**2.)/
        (2.*std_mountain**2.)
        )
    return H


       
#     elif orography == EARTH_OROGRAPHY:
#        mat_contents = sio.loadmat('digital_elevation_map.mat')
#        H = mat_contents['elevation'];
#        # Enforce periodic boundary conditions in x
#        H[0, :] = H[-2, :]
#        H[-1, :] = H[1, :]
# #       H[[0, -1],:]=H[[-2, 1],:];
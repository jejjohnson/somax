import math

R_EARTH = 1000 * (6357 + 6378) / 2  # radius of the Earth (m)
GRAVITY = 9.80665  # gravitational acceleration (m/s^2)
OMEGA = (
    2.0 * math.pi / 86_164.0
)  # angular speed of the Earth (7.292e-5) (rad/s)
DEG2M = 2 * math.pi * R_EARTH / 360.0  # Degrees to Meters
RHO = 1.0e3  # density of water (kg/m^3)

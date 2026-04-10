"""Human-readable formatters for units, used in run-log output.

Centralizes the unit-pretty-print rules so the diagnostic lines in
``run.log`` are legible at a glance instead of being a wall of
scientific notation.
"""

from __future__ import annotations


# Field name -> SI unit string. Covers the State field names from the
# models registered in ``gfd_testcases.py``. Unknown fields fall back to
# no unit annotation.
FIELD_UNITS: dict[str, str] = {
    "h": "m",
    "u": "m/s",
    "v": "m/s",
    "q": "1/s",  # PV (relative vorticity-like)
    "psi": "m²/s",
    "xyz": "",  # Lorenz '63
    "x": "",  # Lorenz '96
}


# Time conversion thresholds: (cutoff_seconds, unit_label, divisor).
# First match wins.
_TIME_THRESHOLDS: list[tuple[float, str, float]] = [
    (60.0, "s", 1.0),
    (3600.0, "min", 60.0),
    (86400.0, "hr", 3600.0),
    (2_592_000.0, "day", 86400.0),
    (31_557_600.0, "month", 2_592_000.0),
    (float("inf"), "yr", 31_557_600.0),
]


def format_time_seconds(seconds: float) -> str:
    """Format a time in seconds with a parenthesized human-readable unit.

    Examples:
        ``86400`` → ``"86400 s (1.0 day)"``
        ``31557600`` → ``"3.156e+07 s (1.0 yr)"``
        ``42`` → ``"42 s"``
        ``600`` → ``"600 s (10.0 min)"``
    """
    if seconds < 60.0:
        # Sub-minute: just seconds, no parenthetical.
        if seconds == int(seconds):
            return f"{int(seconds)} s"
        return f"{seconds:.3g} s"
    for cutoff, label, divisor in _TIME_THRESHOLDS:
        if seconds < cutoff:
            converted = seconds / divisor
            return f"{seconds:.3g} s ({converted:.2f} {label})"
    # unreachable — last entry has float('inf') cutoff
    return f"{seconds:.3g} s"


def format_distance_meters(meters: float) -> str:
    """Format a distance in meters with a parenthesized km value if useful."""
    if abs(meters) < 1000.0:
        return f"{meters:.3g} m"
    return f"{meters:.3g} m ({meters / 1000.0:.2f} km)"


def format_field_stats(
    field_name: str,
    *,
    min_val: float,
    mean_val: float,
    max_val: float,
    nan_count: int,
) -> str:
    """Format one State field's [min, mean, max] reduction with units.

    The output is short enough to fit several fields on one heartbeat
    line. Example::

        h[m]=[500,2.50e+03,4.62e+03] u[m/s]=[-0.50,1.3e-04,0.50]

    NaN counts are appended only when present.
    """
    unit = FIELD_UNITS.get(field_name, "")
    unit_tag = f"[{unit}]" if unit else ""
    body = f"[{min_val:.3g},{mean_val:.3g},{max_val:.3g}]"
    if nan_count:
        body += f" NaN={nan_count}"
    return f"{field_name}{unit_tag}={body}"


def format_wallclock(seconds: float) -> str:
    """Wallclock formatter (different from sim time — we always show seconds)."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m{secs:.0f}s"
    hours, mins = divmod(minutes, 60)
    return f"{int(hours)}h{int(mins)}m{secs:.0f}s"

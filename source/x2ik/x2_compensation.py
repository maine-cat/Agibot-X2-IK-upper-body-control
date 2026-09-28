"""Current customer-interface compensation; no per-robot calibration lookup.

These are fixed operating parameters, not a measured calibration for every robot.
Legacy engineering/calibration tools retain their separate configuration paths.
"""


def fixed_compensation() -> dict:
    """Return an independent copy of the MDI / public MoveJ parameters."""
    return dict(stiffness=40.0, bias_limit_deg=12.0, gravity_source="pelvis")

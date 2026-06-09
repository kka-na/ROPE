import math

_TAU_MAX = 300.0   # ms
_V_MIN   = 0.5     # m/s

class DRCompensator:
    def __init__(self, enabled=False, target='both'):
        self.enabled = enabled
        self.target  = target  # 'both' | 'v2x_only' | 'lidar_only'

    def compensate(self, x, y, v, heading_deg, tau_ms, source):
        """Linear DR extrapolation.
        Returns (x_c, y_c, applied: bool, fallback_reason: str|None)
        source: 'v2x' | 'lidar'
        """
        if not self.enabled or (self.target != 'both' and self.target != f'{source}_only'):
            return x, y, False, 'disabled'
        try:
            if any(math.isnan(val) for val in [x, y, v, heading_deg, tau_ms]):
                return x, y, False, 'nan_input'
        except TypeError:
            return x, y, False, 'nan_input'
        if tau_ms > _TAU_MAX:
            return x, y, False, 'tau_exceeds_max'
        if abs(v) < _V_MIN:
            return x, y, False, 'low_speed'
        t = tau_ms / 1000.0
        h = math.radians(heading_deg)
        return x + v * math.cos(h) * t, y + v * math.sin(h) * t, True, None

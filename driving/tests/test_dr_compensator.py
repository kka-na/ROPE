import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sharing_info'))
from dr_compensator import DRCompensator

def test_basic_compensation():
    # AC §7.1: px=0, vx=10 m/s, τ=100ms → px_c=1.0
    dr = DRCompensator(enabled=True, target='both')
    x_c, y_c, applied, reason = dr.compensate(0, 0, 10, 0, 100, 'v2x')
    assert abs(x_c - 1.0) < 1e-9, f"expected x_c=1.0, got {x_c}"
    assert abs(y_c) < 1e-9
    assert applied and reason is None

def test_tau_exceeds_max():
    # AC §7.2: τ=500ms → raw pass-through, reason='tau_exceeds_max'
    dr = DRCompensator(enabled=True, target='both')
    x_c, y_c, applied, reason = dr.compensate(5, 5, 10, 0, 500, 'v2x')
    assert (x_c, y_c) == (5, 5)
    assert not applied and reason == 'tau_exceeds_max'

def test_low_speed_fallback():
    # AC §7.3: v=0.1 m/s → no compensation
    dr = DRCompensator(enabled=True, target='both')
    x_c, y_c, applied, reason = dr.compensate(5, 5, 0.1, 0, 100, 'lidar')
    assert (x_c, y_c) == (5, 5)
    assert not applied and reason == 'low_speed'

def test_disabled():
    dr = DRCompensator(enabled=False)
    x_c, y_c, applied, _ = dr.compensate(0, 0, 10, 0, 100, 'v2x')
    assert (x_c, y_c) == (0, 0) and not applied

def test_nan_guard():
    import math
    dr = DRCompensator(enabled=True, target='both')
    x_c, y_c, applied, reason = dr.compensate(float('nan'), 0, 10, 0, 100, 'v2x')
    assert not applied and reason == 'nan_input'

def test_target_filter():
    dr = DRCompensator(enabled=True, target='v2x_only')
    _, _, applied, _ = dr.compensate(0, 0, 10, 0, 100, 'lidar')
    assert not applied

if __name__ == '__main__':
    test_basic_compensation()
    test_tau_exceeds_max()
    test_low_speed_fallback()
    test_disabled()
    test_nan_guard()
    test_target_filter()
    print("All 6 tests passed.")

import numpy as np

from QATCH.processors.CurveOptimizer import estimate_step_delta


def test_estimate_step_delta_clamps_out_of_bounds_slope():
    times = np.arange(12, dtype=float)
    ysm = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 11.0, 22.0, 33.0],
        dtype=float,
    )
    raw = ysm.copy()

    delta = estimate_step_delta(times, raw, ysm, core_start=8, core_end=8, left_bound=0)

    left_slope = np.polyfit(times[:5], ysm[:5], 1)[0]
    right_slope = np.polyfit(times[8:], ysm[8:], 1)[0]
    expected_delta = 0.5 * (left_slope + right_slope) * (times[8] - times[4])

    assert np.isclose(delta, expected_delta)
    assert delta > 0
    assert (
        min(left_slope, right_slope)
        <= delta / (times[8] - times[4])
        <= max(left_slope, right_slope)
    )

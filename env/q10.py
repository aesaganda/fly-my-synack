"""Temperature scaling of neural time constants.

Drosophila are ectotherms: membrane and synaptic kinetics track ambient
temperature rather than being held at a regulated set point. The standard
phenomenological way to express that is a Q10 coefficient - the factor by which
a rate changes per 10 C.

    tau(T) = tau_ref * q10 ** ((T_ref - T) / 10)

so a HIGHER temperature gives a SMALLER tau (faster dynamics).

The q10 value itself is a PLACEHOLDER supplied by the preset YAML, not a
constant sourced from the literature. Reported Q10 values for neural processes
generally sit somewhere around 2-3, but the right number depends on which
process you mean (membrane time constant, synaptic decay, spike threshold
dynamics, muscle activation) and none of those is calibrated here. Treat the
YAML value as a knob to fit, not a fact to cite.
"""

from __future__ import annotations

# Reference temperature the *unscaled* tau values are defined at.
# 25 C, matching the dry_land baseline preset.
T_REF_C = 25.0


def q10_factor(temperature_c: float, q10: float, t_ref_c: float = T_REF_C) -> float:
    """Multiplicative factor applied to a time constant at `temperature_c`.

    >>> round(q10_factor(25.0, 2.3), 6)
    1.0
    >>> q10_factor(35.0, 2.3) < 1.0   # hotter -> faster -> smaller tau
    True
    >>> q10_factor(15.0, 2.3) > 1.0   # colder -> slower -> larger tau
    True
    """
    if q10 <= 0:
        raise ValueError(f"q10 must be positive, got {q10!r}")
    return float(q10 ** ((t_ref_c - temperature_c) / 10.0))


def scale_tau_ms(
    tau_ms: float,
    temperature_c: float,
    q10: float,
    clamp_ms: tuple[float, float],
    t_ref_c: float = T_REF_C,
) -> float:
    """Scale one time constant and clamp it to a plausible range.

    The clamp is a guard rail, not biology: without it an extreme preset
    temperature can drive tau to zero and blow up the LIF integration.
    """
    lo, hi = clamp_ms
    if lo > hi:
        raise ValueError(f"tau_clamp_ms is inverted: {clamp_ms!r}")
    return min(max(tau_ms * q10_factor(temperature_c, q10, t_ref_c), lo), hi)

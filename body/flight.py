"""Wingbeat and flight forces.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Real: the wings. They are articulated bodies on the NeuroMechFly model with
three DoFs each, they are driven at a *Drosophila* wingbeat frequency, and they
beat with a feathering flip at stroke reversal. What you see is a wingbeat.

Not real: the forces. **The wings do not lift this fly and cannot.** That is
measured, not assumed. With MuJoCo's per-geom ellipsoid fluid model enabled on
the wings, a full 6.6 mm feathered stroke at 200 Hz was swept across every
feather phase from 0 to 360 deg and both plausible stroke axes:

    best net vertical force found:  0.072 x body weight
    most phases:                   |0.03| x body weight, either sign

Insect lift comes from unsteady mechanisms - the leading-edge vortex, rotational
circulation, wake capture - and a quasi-steady fluid model has no representation
of any of them. No amount of tuning gets 1.0 out of 0.07.

So the aerodynamics are lumped into a controller that applies a wrench to the
thorax, in exactly the spirit of the two approximations this project already
makes: the CPG generates the gait because the connectome cannot, and buoyancy is
folded into gravity because MuJoCo has none. The wings are animated; the
controller flies the fly.

The controller commands VELOCITIES rather than forces, which is both easier to
steer and closer to what the animal does: a fly holds airspeed and attitude with
haltere and visual feedback rather than by setting muscle forces open-loop. The
attitude and yaw-rate loops here stand in for the halteres specifically - the
model has them, they are mechanosensory rate gyros, and this is the one job they
do.

`body/` never imports torch or anything from `brain/`/`bridge/`.
"""

from __future__ import annotations

import math

import numpy as np

# ---- the wingbeat --------------------------------------------------------
# 200 Hz is the real thing for Drosophila. At the 1e-4 s timestep that is 50
# steps a beat, which is coarse but adequate for something that is animated
# rather than integrated for force. It also means the wings are a blur in any
# rendered frame, which is what a fly looks like.
WINGBEAT_HZ = 200.0
# Stroke on the wing's yaw axis, which is the one that moves the wing up and
# down - measured on the model, +0.6 rad moves the wing centre 0.37 mm in z
# against 0.05 mm for pitch.
STROKE_RAD = 1.1
# Feathering on the pitch axis, the one that turns the wing over: +0.6 rad
# swings the wing normal 23 deg. The tanh makes the flip fast and the rest of
# the stroke flat, which is what a wing actually does.
FEATHER_RAD = 1.2
FEATHER_SHARPNESS = 4.0

# The wing joints ship with damping 0.5 and stiffness 10, which are sensible for
# a wing that is folded away and hopeless for one that beats: the resulting
# actuator bandwidth is kp/damping = 800 rad/s, so at 200 Hz the stroke came out
# at 0.18 mm instead of 6.6 mm. Measured.
WING_KP = 2000.0
WING_DAMPING = 0.05

# ---- the controller ------------------------------------------------------
# Force per unit of velocity error, in body weights per (mm/s). 0.03 gives a
# ~30 ms velocity time constant, which settles without overshoot at this
# timestep.
K_VELOCITY = 0.030
# ...and a ceiling on how much of it can be asked for at once, in body weights.
# Without this a 250 mm/s velocity error commands 7.5 weights of thrust, which
# is not a manoeuvre any animal makes and is violently unstable: measured, the
# fly was slammed forward, tipped, and drove itself into the floor from 200 mm
# within a second. Real Drosophila manage roughly 1-2 g in a saccade.
MAX_MANOEUVRE_WEIGHTS = 1.2
# Attitude: torque per unit of sin(tilt). Sized against the fly's rotational
# inertia (~4.6e-4 g mm^2): at 0.2 and below it tumbles - measured, peak tilt
# 176 deg, i.e. fully inverted - and at 6.0 it holds inside 6 deg while hovering.
# The damping has to be near critical for the fly's rotational inertia
# (~4.6e-4 g mm^2), which is 2*sqrt(K*I). At a quarter of that it looked stable
# in a hover started from level and tumbled the moment it took off from the
# ground with any tilt at all: peak roll +/-130 deg at 400 rad/s. Measured on
# takeoff, which is the hard case:
#     K=1.5  D=0.05   tumbles, tilt 178 deg
#     K=6    D=0.21   tilt 45 deg, drifts 355 mm
#     K=20   D=0.38   tilt 16 deg, drifts 93 mm
#     K=60   D=0.66   tilt 10 deg, drifts 30 mm    <- chosen
#     K=150  D=1.05   tilt  9 deg, drifts 11 mm, but stiff enough to fight the
#                     wingbeat and buy nothing
K_ATTITUDE = 60.0
D_ATTITUDE = 0.66
# Yaw is a RATE loop, not an attitude loop: a fly has no preferred heading, and
# the beating wings leave a small residual yaw torque that a pure damping term
# cannot cancel (measured: 30 deg/s of drift in hover).
K_YAW_RATE = 0.02

# Lift is divided by the vertical component of the body's up axis so that the
# vertical force is what was asked for whatever the fly's attitude. Clamped, or
# a fly knocked past 70 deg would command unbounded lift trying to recover.
MIN_UP_COMPONENT = 0.35


def wing_angles(t: float, amplitude: float = 1.0) -> tuple[float, float]:
    """(stroke, feather) joint angles for both wings at time `t`.

    Both wings get the same angles: the wing joint axes are NOT mirrored on this
    model, unlike the legs - verified, +0.8 rad of yaw moves both wing tips down
    by the same 1.76 mm.
    """
    phase = 2.0 * np.pi * WINGBEAT_HZ * t
    stroke = amplitude * STROKE_RAD * np.sin(phase)
    feather = amplitude * FEATHER_RAD * np.tanh(FEATHER_SHARPNESS * np.cos(phase))
    return float(stroke), float(feather)


def body_wrench(
    rotation: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    weight: float,
    climb_mm_s: float,
    forward_mm_s: float,
    yaw_rate: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Force and torque to apply to the thorax, as plain 3-tuples.

    rotation:  3x3 body-to-world matrix. Columns are the body's forward, left
               and up axes.
    velocity:  world linear velocity of the body, mm/s.
    weight:    the fly's weight in model force units, so the gains above are in
               body weights and do not care what the fly masses.

    Written out in scalars rather than as NumPy vector expressions. This runs
    once per PHYSICS step - holding it at the control rate makes the attitude
    loop unstable, measured - and at that rate NumPy's per-call overhead on
    three-element arrays is the entire cost: 50 us a step against about 8 here,
    which was most of the budget once `mj_step` itself is only 68 us.
    """
    fx, fy, fz = rotation[0, 0], rotation[1, 0], rotation[2, 0]      # forward
    lx, ly, lz = rotation[0, 1], rotation[1, 1], rotation[2, 1]      # left
    ux, uy, uz = rotation[0, 2], rotation[1, 2], rotation[2, 2]      # up
    vx, vy, vz = velocity[0], velocity[1], velocity[2]
    wx, wy, wz = angular_velocity[0], angular_velocity[1], angular_velocity[2]

    # Hold the fly up. Dividing by the vertical component of `up` keeps the
    # VERTICAL force equal to the weight whatever the attitude; clamped, or a
    # fly knocked past 70 deg would command unbounded lift trying to recover.
    support = weight / max(uz, MIN_UP_COMPONENT)
    sx, sy, sz = ux * support, uy * support, uz * support

    # Correct velocity error along three body axes. The sideways term is what
    # stops a hovering fly sliding away; a real one holds station optically.
    gain = weight * K_VELOCITY
    along_forward = forward_mm_s - (vx * fx + vy * fy + vz * fz)
    along_left = -(vx * lx + vy * ly + vz * lz)
    climb_error = climb_mm_s - vz
    mx = (fx * along_forward + lx * along_left) * gain
    my = (fy * along_forward + ly * along_left) * gain
    mz = (fz * along_forward + lz * along_left + climb_error) * gain

    ceiling = weight * MAX_MANOEUVRE_WEIGHTS
    magnitude = math.sqrt(mx * mx + my * my + mz * mz)
    if magnitude > ceiling:
        scale = ceiling / magnitude
        mx, my, mz = mx * scale, my * scale, mz * scale

    # Level the body, damp the roll and pitch rates, and drive yaw to a rate.
    #
    # The damping is applied ONLY to the tilt component. Damping the full
    # angular velocity damps yaw too, and at these gains that swamps the yaw
    # command completely - measured, a commanded 1.5 rad/s turn produced a path
    # indistinguishable from flying straight.
    yaw_component = wx * ux + wy * uy + wz * uz
    yaw_torque = K_YAW_RATE * (yaw_rate - yaw_component)
    # cross(up, world-up) = (uy, -ux, 0)
    tx = K_ATTITUDE * uy - D_ATTITUDE * (wx - yaw_component * ux) + yaw_torque * ux
    ty = -K_ATTITUDE * ux - D_ATTITUDE * (wy - yaw_component * uy) + yaw_torque * uy
    tz = -D_ATTITUDE * (wz - yaw_component * uz) + yaw_torque * uz
    return (sx + mx, sy + my, sz + mz), (tx, ty, tz)

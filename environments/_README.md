# Environment presets

One YAML per preset. These are **data** — adding a preset means adding a file
here, never editing Python.

## Units: mm / g / s  (NOT SI)

FlyGym's NeuroMechFly model works in millimetres, grams and seconds
(gravity is `-9810`, thorax mass `3.07e-4`). Fluid values pasted from an SI
table will be wrong by up to six orders of magnitude.

| field      | unit       | air @25C  | water @20C |
|------------|------------|-----------|------------|
| `density`  | g/mm^3     | 1.184e-6  | 9.98e-4    |
| `viscosity`| g/(mm*s)   | 1.84e-5   | 1.002e-3   |
| `wind`     | mm/s       | 500 = 0.5 m/s        |
| `gravity`  | mm/s^2     | -9810                |

`units: mm_g_s` is mandatory in every preset; the loader rejects anything else
rather than silently simulating a fly in treacle.

## Two MuJoCo gotchas encoded here

1. `density: 0` disables lift/drag and `viscosity: 0` disables viscous forces.
   FlyGym ships both unset, so fluid forces are OFF by default.
2. `wind` on its own does nothing. MuJoCo subtracts it from body velocity and
   scales the result by density/viscosity, so a wind preset with zero density
   is a no-op. `tests/test_env_loader.py` asserts this.

## Placeholders

`q10`, `tau_clamp_ms`, and everything under `sensory:` are **not** sourced
biological constants. They are tunable starting points, marked in each file.
See the "Approximations" table in the README.

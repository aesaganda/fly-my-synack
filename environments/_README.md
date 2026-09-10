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

## Optional physics fields

Beyond the required ones, a preset may set:

| field | default | meaning |
|---|---|---|
| `adhesion_gain` | 200 | tarsal grip - see the humidity curve below. |
| `locomotion` | `walk` | `walk` (tripod gait, adhesion on) or `swim` (synchronous rowing, adhesion off). |
| `stroke_freq_hz` | body default (18) | limb cycle frequency. Swimming wants a slower one. |

## Optional sensory fields

| field | default | meaning |
|---|---|---|
| `olfactory_gain` | 1.0 | how well the fly can smell here. Only does anything in a world that has odour sources — `--world kitchen`. |

`olfactory_gain` is the environment's grip on the fly's *exploration*, the way
`adhesion_gain` is its grip on the fly's *feet*. The `windy` preset cuts it to
0.45 for a specific reason: the plume model in `body/world.py` is a still-air
Gaussian, and wind does not merely translate a plume, it shreds it into
filaments that a smooth spatial gradient does not describe. Lowering the gain
says "the cue is unreliable here" instead of pretending the model still holds.
`submerged_water` cuts it to 0.15 because airborne olfaction underwater is not
a thing. All of these numbers are PLACEHOLDERS.

## Grip is not monotonic in humidity

Insect tarsal attachment is maximal at INTERMEDIATE humidity. The adhesive pads
need some moisture to form the capillary bridges that produce grip, so very dry
air weakens them; a condensed water film on the substrate makes them slip, so
very wet air weakens them too. The three air presets sit on that curve:

| preset | RH | `adhesion_gain` | why |
|---|---|---|---|
| `dry_land` | ~20% | 120 | pads dried out, few capillary bridges |
| `temperate` | ~60% | 200 | optimum - the reference |
| `humid_air` | ~90% | 80 | water film on the surface |

So the fastest walking is in the MIDDLE of the humidity range, not at either
end. The shape is documented in the literature; the specific numbers here are
placeholders.

## Placeholders

`q10`, `tau_clamp_ms`, and everything under `sensory:` are **not** sourced
biological constants. They are tunable starting points, marked in each file.
See the "Approximations" table in the README.

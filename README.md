# Connectome-driven fly simulation

A biomechanical *Drosophila* body (FlyGym / NeuroMechFly v2 on MuJoCo) driven by
a sparse spiking network whose **connectivity** comes from the Janelia male CNS
connectome (`male-cns:v1.0`), runnable across seven environment presets that
perturb both the physics and the neural dynamics — on bare ground, on a kitchen
worktop under a Matrix sky, or in a room it explores by smell and crosses on the
wing.

> **Read this first.** This is a modelling toy built on real connectivity. It is
> not biophysics and it is not a validated model of anything. The
> [What is real and what is not](#what-is-real-and-what-is-not) table is the
> most important section in this file — please don't cite the pretty numbers
> without it.

---

## Quick start

### Without Docker

`run.sh` builds the whole stack into a local venv and runs it. It creates the
venv on first use, so this works on a fresh checkout:

```bash
./run.sh test                         # build if needed, then run the tests
./run.sh sim --env submerged_water --steps 20000
./run.sh sim --world kitchen --steps 40000   # a worktop, under a Matrix sky
./run.sh sim --world room --steps 200000     # indoors, where it flies
./run.sh compare dry_land submerged_water windy
./run.sh web                          # http://localhost:8000 - kitchen by default
```

Needs Python 3.12–3.14 (`brew install python@3.12`, or
`apt install python3.12 python3.12-venv`). Behind a package mirror:

```bash
PIP_INDEX_URL=https://your-mirror/artifactory/api/pypi/pypi/simple ./run.sh setup
```

It also exports the macOS system CA bundle automatically, so neuPrint fetches
work behind a TLS-inspecting proxy without extra setup.

### With Docker

No credentials and no network needed:

```bash
docker compose build sim
docker compose run --rm sim python3 run.py --env dry_land --steps 5000 --connectome synthetic
```

The scientific payoff — one brain, several environments:

```bash
docker compose run --rm sim python3 run.py \
  --compare-envs dry_land submerged_water windy --steps 20000
```

Browser UI on <http://localhost:8080> — the kitchen world by default:

```bash
docker compose up web
```

Set `WEB_PORT` if 8080 is taken, and `FLY_WORLD=flat` for bare ground.

---

## What it does

```
CONNECTOME  (neuPrint male-cns:v1.0)
  -> sparse LIF network (PyTorch, CPU or CUDA)
  -> descending neurons + VNC motor neurons as the output layer
       |
BRIDGE   decode MN/DN population rates -> descending drive
         encode contact + air motion + SMELL -> injected current
       |
BODY     tripod CPG -> 42 leg joint targets -> MuJoCo
         wingbeat + flight controller -> a wrench on the thorax
ENV      preset YAML -> MuJoCo fluid options AND LIF time constants
WORLD    scenery, collision, and where the smells are
       |
  sensory feedback -> spikes -> back into the network
```

**The connectome does not generate walking.** A connectome gives connectivity,
not tuned synaptic weights, so a LIF network built from it will not produce a
gait. Here a central pattern generator produces the tripod gait and the decoded
neural activity *steers* it (speed, turn, per-leg gain, stop). That is a
documented hybrid, chosen deliberately over a "pure" version that would only
flail.

---

## Environment presets

`environments/*.yaml`, one file per preset — data, not code. Adding
`environments/vacuum.yaml` makes `--env vacuum` work with no code change.

| preset | medium | T | what changes |
|---|---|---|---|
| `temperate` | still air, ~60% RH | 25 °C | reference — best grip |
| `dry_land` | still air, ~20% RH | 25 °C | dried pads, weaker grip |
| `humid_air` | near-saturated air | 25 °C | water film, weakest grip |
| `submerged_water` | water | 20 °C | ~1000× density, `implicitfast` integrator |
| `hot` | dry air | 35 °C | past the thermal optimum — frantic |
| `cold` | dry air | 10 °C | near chill coma — sags and drags |
| `windy` | dry air, gusty 0→1.9 m/s | 25 °C | buffeted off course |

```bash
python3 run.py --list-envs
```

### ⚠️ Units are mm / g / s, not SI

NeuroMechFly works in millimetres, grams and seconds — gravity is `-9810`, the
thorax weighs `3.07e-4`. **SI fluid values are wrong by up to a factor of a
million.**

| field | unit | air @25 °C | water @20 °C | vs SI |
|---|---|---|---|---|
| `density` | g/mm³ | `1.184e-6` | `9.98e-4` | ×1e-6 |
| `viscosity` | g/(mm·s) | `1.84e-5` | `1.002e-3` | ×1 (unchanged) |
| `wind` | mm/s | `900` mean + `1200` gust | — | ×1e3 |
| `gravity` | mm/s² | `-9810` | | ×1e3 |

Every preset declares `units: mm_g_s` and the loader rejects anything else.
`tests/test_env_loader.py` asserts water's density is ~1e-3 and not ~1000.

### A Docker trap, for the record

`docker compose build` did not work at all until this was found, and the failure
mode is worth knowing about generally: **an explicitly empty `--build-arg`
overrides an `ARG` default.** `docker-compose.yml` passes
`TORCH_INDEX: ${TORCH_INDEX:-}`, which is an empty string rather than "unset",
so `ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu` in the Dockerfile was
silently replaced by `""` and the build ran `pip install --index-url ""`. The
default now lives in the `RUN` (`${TORCH_INDEX:-$TORCH_CPU_INDEX}`), where an
empty value falls through as intended.

Worth the fix twice over: the CPU-only index has wheels for aarch64 as well as
x86_64, and using it takes the image from **7.89 GB to 3.18 GB** — the ordinary
PyPI torch wheel bundles CUDA on both architectures.

### Two MuJoCo traps this encodes

1. `density: 0` disables lift/drag and `viscosity: 0` disables viscous forces.
   FlyGym ships both unset, so **fluid forces are off by default**.
2. **`wind` alone does nothing.** MuJoCo subtracts it from body velocity and
   scales the result by density/viscosity, so a windy preset with zero density
   is a silent no-op. The loader raises rather than let that through.

MuJoCo also warns that the Euler integrator handles body viscosity poorly, and
FlyGym defaults to Euler — so `submerged_water` sets `integrator: implicitfast`.
That is a YAML field, not a hardcoded special case.

---

## Worlds: bare ground, or a kitchen under a Matrix sky

A **preset** is the medium the fly is in; a **world** is the place it is in.
They are orthogonal — a preset writes to `mjOption`, a world writes geometry —
so `--world kitchen --env windy` is a gusty kitchen and every preset still
works in both.

| `--world` | what it is | default for |
|---|---|---|
| `flat` | FlyGym's bare checkerboard plane. No scenery, no smells, no wings. | `run.py` |
| `kitchen` | a worktop with crumbs, spills and crockery on it, under an animated digital-rain sky, and odour sources the fly can follow | — |
| `room` | the same worktop, indoors: walls, ceiling, counter, table, fridge, a lamp, a window with the rain outside — **and air space, so the fly flies** | the browser UI (`FLY_WORLD`) |

`flat` stays the CLI default deliberately: every kinematics number in this
README was measured on bare ground, and the kitchen's smells steer the fly, so
switching the default would silently invalidate them.

The kitchen is a hand-placed vignette about 150 mm across — sugar, jam, crumbs,
a chopping board, crockery, and the furniture on the horizon — plus 300 more
crumbs and spills generated from a fixed seed out to 620 mm. That second part is
not decoration: the fly nets ~9 mm/s, so it walks out of a hand-placed scene in
under half a minute, and a browser demo runs for as long as the tab is open.

It costs about 15% of the step rate — 1,075 against 1,271 steps/s on the dev
machine, for 330 extra geoms, 80 extra collision pairs and an odour lookup at
two antennae every step.

### The sky

MuJoCo compiles a skybox as six square faces stacked vertically in one texture
(right, left, up, down, front, back, row 0 at the top — verified by rendering a
gradient, not by trusting the docs). `body/world.py` generates green rain into
that texture and re-uploads it to the GL context with `mjr_uploadTexture` on
every rendered frame. Nothing is recompiled, and the phase comes from
`mj_data.time` rather than a wall clock, so a run renders identically twice.

One trap worth naming: FlyGym ships `zfar = 250` (× the unit model extent, so
250 mm), which **silently deletes** anything further away. The first version of
this scene had a mug and a toaster on the horizon that simply never rendered.
`KitchenWorld.apply_visuals` pushes the far plane out, brings `znear` up with it
so the depth buffer does not span 0.5 µm to 1.2 m, and turns on haze so the
ground fades into the sky instead of ending at a hard edge.

### What "exploring" means here

**It is not consciousness and the code does not claim to be.** It is two small
mechanisms in a closed loop, and both are there because removing them produces
a specific, measured failure. Two further mechanisms were built, measured, and
deleted for losing to the empty control — see
[A failure worth recording](#a-failure-worth-recording).

1. **Bilateral olfaction.** Concentration is sampled at the two antennae — the
   third antennal segments, where the olfactory receptor neurons actually are —
   and injected into the **left and right descending populations**. That is the
   whole trick: `MotorDecoder` already reads its turn command as
   `tanh((DN_R − DN_L) / ref)`, so a smell on the right raises DN_R and the fly
   turns right, with no new decode path. Sign errors here produce a fly that
   flees food, which looks like plausible behaviour, so
   `tests/test_world.py::test_a_smell_on_the_right_produces_a_right_turn` pins
   it.

2. **Receptor adaptation.** A leaky baseline per antenna, subtracted before the
   drive is computed. Without it the fly orbits the first source it finds for
   ever — at the peak of a plume the gradient reverses on every pass and pulls
   it straight back in. Adaptation is what lets it arrive, lose interest, and go
   somewhere else; it is the mechanism that turns chemotaxis into exploration.

That is the whole of it. The two together produce the behaviour: over 150k
steps the fly leaves the spawn, tracks in on the sugar, habituates, wanders off,
finds the cake, and stays inside about 116 mm of where it started, with a path
straightness of 0.77 — a genuinely curved, looping trajectory rather than a walk
across the frame. Nothing schedules any of that.

Read the live values in the browser UI's **Exploring** panel, or in
`Session.live_metrics()`: `odour`, `odour_lr_diff`, `nearest_source`.

`--world flat` gets none of it, and it was bit-identical to the old code until
the SciPy matmul landed. It still is *numerically* the same model — SciPy simply
accumulates a float32 sum in a different order, which over 12,000 steps of a
chaotic system diverges to about 1 part in 1e7 on `dry_land` and 0.1% on
`windy`. `tests/test_world.py::test_flat_world_gets_none_of_the_new_machinery`
pins the conditions that keep the *mechanism* out of the flat world.

Environments reach this too. `olfactory_gain` in a preset says how well the fly
can smell there — `windy` cuts it to 0.45 because a still-air Gaussian plume is
not a description of wind-shredded filaments, and `submerged_water` to 0.15
because airborne olfaction underwater is not a thing. See
`environments/_README.md`.

### Flight

`--world room` gives the fly wings and somewhere to use them. It walks, tracks a
smell, habituates, takes off, crosses the room, and lands on the next thing it
smells — about four takeoffs a minute of simulated time.

**The wings are real and they do not lift the fly.** Both halves matter:

* Real: the wings are articulated bodies on the NeuroMechFly model with three
  DoFs each. They beat at 200 Hz — a *Drosophila* wingbeat — with a feathering
  flip at stroke reversal, driven by position actuators. What you see is a
  wingbeat.
* Not real: the forces. With MuJoCo's per-geom ellipsoid fluid model on the
  wings, a full 6.6 mm feathered stroke at 200 Hz was swept across **every
  feather phase from 0 to 360°** and both plausible stroke axes. The best net
  vertical force found anywhere in that sweep was **0.072 × body weight**; most
  phases gave |0.03| in either direction. Insect lift comes from the
  leading-edge vortex, rotational circulation and wake capture, and a
  quasi-steady fluid model has none of them. No amount of tuning gets 1.0 out of
  0.07.

So the aerodynamics are lumped into a controller that applies a wrench to the
thorax — the same kind of approximation as the CPG generating the gait, and as
buoyancy being folded into gravity. It commands *velocities*, not forces, which
is both easier to steer and closer to what the animal does: a fly holds airspeed
and attitude with haltere and visual feedback, not by setting muscle forces
open-loop. The attitude and yaw-rate loops stand in for the halteres
specifically — the model has them, they are mechanosensory rate gyros, and this
is the one job they do.

The connectome steers it exactly as it steers walking: decoded `forward` sets
cruise speed, decoded `turn` sets yaw rate. Takeoff and landing come from the
olfactory loop already there — receptor adaptation decides when the fly is
*finished* with a smell, and leaving is what an animal does next.

Three things needed a stand-in that is not connectome, all listed in the
approximations table: airborne arousal (the same "no ground contact, no limb
load, no drive" torpor that `submerged_water` documents), a contact-avoidance
reflex, and knowing where the walls are.

### The world has to be walkable

This gait cannot climb and cannot reverse, and both limits are sharper than
they look:

* a wall **0.4 mm** high — half the fly's standing height — stops it dead;
* running the CPG phase backwards does not walk it backwards, it walks it
  *sideways*: the duty-factor asymmetry means the step cycle is not
  time-reversible.

So anything wide enough that the fly cannot slide off the end of it is a
permanent trap. Only props small enough to slide past are given collision
pairs — measured, the fly clears a 3.6 mm cube after about 13k steps of contact
and walks away. Plates, boards and the far-off furniture are scenery it passes
through; at the camera's 9 mm standoff that is invisible, and a plate rim the
fly can never escape is not.

---

## Real time

**The simulation runs at real time.** Measured on an M3 Pro as steps per second
of *process CPU time*, five repetitions, median, `--world room`:

| | steps/s | x real time |
|---|---|---|
| walking, `--control-every 1` *(the documented model)* | 3,724 | 0.37 |
| walking, `--control-every 30` | 14,113 | **1.41** |
| flying, `--control-every 30` | 14,174 | **1.42** |

CPU time rather than wall time on purpose — see
[measuring on a busy machine](#measuring-on-a-busy-machine) below, which is the
most useful thing in this section.

The **browser** gets that multiplied by the fraction of the clock left after
rendering: a frame costs 10-12 ms to draw and JPEG-encode, so at the default
50 ms stepping budget it keeps about 80%. The header reports what it actually
achieved rather than claiming a figure.

Getting even this far took finding out which of three obvious levers worked.

### The physics timestep cannot move

The first idea is to take bigger steps: real time is `1/dt` steps a second, so a
coarser `dt` is a linear win. It is also the one thing this model cannot afford.
Measured on `dry_land` in the kitchen:

| timestep | x real time | net speed mm/s | slip | straightness | fell over |
|---|---|---|---|---|---|
| **1e-4** *(default)* | 0.35 | **8.57** | 0.60 | 0.97 | no |
| 2e-4 | 0.44 | 26.03 | −0.09 | 0.93 | no |
| 3e-4 | 0.77 | 1.18 | 0.61 | 0.42 | **yes** |
| 4e-4 | 0.82 | 4.87 | −0.52 | 0.96 | **yes** |
| 5e-4 | 1.37 | 2.82 | 0.54 | 0.89 | **yes** |

At 2e-4 the fly's walking speed triples and the slip ratio goes negative, which
is not a number that can happen; from 3e-4 it falls over. `--timestep` exists
and is documented as a footgun.

### MuJoCo was never the bottleneck

`mj_step` on its own costs **62–68 µs** — a ceiling of about 15,000 steps a
second, comfortably above the 10,000 real time needs. The full loop was managing
3,400. **Four fifths of the time was Python**, not physics.

### So run the controller slower than the physics

The physics needs 1e-4 for contact stability. The *controller* does not: a
stride is 55 ms long, so updating joint targets every 2 ms still samples it 27
times. `--control-every N` runs the CPG, the brain, the sensory encoding and the
observation readout once per N physics steps, and holds them in between.

| `--control-every` | steps/s | x real time | net speed mm/s | slip | straightness |
|---|---|---|---|---|---|
| **1** *(the documented model)* | 2,521 | 0.25 | 9.17 | 0.58 | 0.98 |
| 2 | 3,627 | 0.36 | 9.11 | 0.59 | 0.98 |
| 5 | 5,463 | 0.55 | 9.38 | 0.57 | 0.98 |
| 10 | 7,618 | 0.76 | 10.27 | 0.54 | 0.98 |
| 20 | 10,673 | **1.07** | 8.62 | 0.60 | 0.88 |
| **30** *(the browser default)* | 10,697 | **1.07** | 6.23 | 0.69 | 0.72 |
| 40 | 12,291 | 1.23 | 2.14 | 0.89 | 0.27 |

The gait is intact to 30 and gone by 40 - by which point the fly is shuffling,
not walking. `run.py` defaults to 1 so the numbers elsewhere in this README stay
reproducible; the web service defaults to 30, which measured 1.24x real time
walking and 1.45x flying on an idle machine.

### Measuring on a busy machine

Every wall-clock number in this section was first measured wrong, and one
conclusion was drawn backwards from it.

The development machine was running several Kubernetes clusters, buildkit
builders and assorted containers in the background: **load average 15+ on twelve
cores, with the container runtime alone taking 551% CPU.** Against that, the
same configuration measured anywhere between 2,225 and 10,603 steps/s on
different runs — a four-fold spread that looked exactly like a real difference
between configurations.

It produced a confident wrong answer. MuJoCo's no-slip pass appeared to be half
the cost of a physics step (183 µs against 88 µs without it), which would have
been worth trading physics fidelity for. Re-measured as **process CPU time with
five repetitions and a median**, dropping it is *slower*: 1.30x against 1.41x.
The whole effect was scheduling noise. The option was written, measured, and
deleted.

If you are benchmarking this, use `time.process_time()`, repeat, and take the
median. Wall-clock timing of a CPU-bound loop on a shared machine measures the
machine, not the code.

### Two things that had to be right

**The sensory loop hung open.** Sensory input is encoded on the control step
just before the network reads it, and the condition for that was written
`(step_count + 1) % brain_every == 0`. Once control steps arrive `N` apart that
condition can never fire — so at any `--control-every` above 1 the network saw
nothing but its tonic drive, the decoded forward drive fell from 0.90 to 0.30,
and the fly shuffled without travelling. It looked exactly like a physical
problem with holding the joint targets. It was `+ 1` where `+ control_every`
belonged.

**The flight wrench genuinely needs the physics rate.** Holding it at the
control rate tips the fly over — the attitude loop is a damper, and sampling a
damper too slowly injects energy. But recomputing it cost 50 µs a step, most of
the substep budget once `mj_step` is only 68. Rewriting `body_wrench` in plain
scalars instead of NumPy three-vectors took it to **12 µs**: at this size NumPy's
per-call overhead *was* the computation.

### What it costs

Frames are still paced on wall time, so a slower machine loses speed rather than
smoothness. `FLY_CONTROL_EVERY=1` restores the documented model in the browser
at about a quarter of real time.

### Apple Metal (MPS)

Supported — `--gpu mps`, or `FLY_DEVICE=mps` for the web service — and **it is
the wrong choice for the default neuron subset.** Measured on an M3 Pro:

| neurons | LIF step, CPU | LIF step, MPS |
|---|---|---|
| 2,129 (`--neuron-subset motor`, the default) | **90 µs** | 294 µs |
| 20,000 (`vnc`) | 1,047 µs | **467 µs** |
| 60,000 | 3,986 µs | **820 µs** |

A LIF step is fourteen elementwise operations against one matrix multiply. Metal
wins the multiply (36 µs against 55 µs at the default size, 4x at 60k neurons)
and loses every elementwise op by ~12 µs each, which at this size is the whole
budget. End to end the default subset runs at 1,588 steps/s on CPU against 567
on MPS. The crossover is somewhere around 20k neurons, so Metal is for running a
*bigger brain*, not for running this one faster.

The same argument applies to CUDA, and is the reason the GPU path was never the
answer to "why is it slow".

---

## Connectome data

Dataset string is exactly **`male-cns:v1.0`** on <https://neuprint.janelia.org>.
It includes the **VNC in the same volume as the brain**, so motor neurons are
present and `manc` is not needed.

| `--connectome` | needs | covers | notes |
|---|---|---|---|
| `neuprint` *(default)* | network | `motor`, `vnc` subsets | works **without a token** |
| `feather` | local files | up to the full 176k network | see below |
| `synthetic` | nothing | any | generated; used by the tests |

### Getting a neuPrint token (optional)

The Cypher endpoint currently answers anonymously, so the default path needs no
credentials. A token raises the rate limits: sign in at
<https://neuprint.janelia.org>, open the account menu → *Auth Token*, and pass it
as `NEUPRINT_TOKEN` (the underlying library's own variable name is
`NEUPRINT_APPLICATION_CREDENTIALS`; both are accepted).

### Full network: manual Feather download

The full connectome only comes from the bulk exports at
<https://male-cns.janelia.org/download/> (CC-BY 4.0). Base URL
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`:

| file | size | purpose |
|---|---|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB | adjacency |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 MB | roles |
| `body-neurotransmitters-male-cns-v1.0.feather` | 42 MB | E/I sign |

Skip the 6.8 GB and 12.7 GB per-synapse files — nothing here uses them. Drop the
files into the `connectome` volume and run with `--connectome feather`.

Column names inside these files could not be verified during development
(Google Cloud Storage is blocked on the development network), so
`brain/sources/feather.py` resolves every column through an alias table and
tells you what it found if a name is missing:

```bash
docker compose run --rm sim python3 -m brain.sources.feather --schema /data/connectome
```

### Neuron subsets

| `--neuron-subset` | neurons | notes |
|---|---|---|
| `motor` *(default)* | ~2.1k | 1,314 DNs + 815 MNs — the layers the bridge reads |
| `vnc` | ~17k | adds VNC interneurons and ascending neurons |
| `all` | ~176k | `feather` only |

---

## What is real and what is not

**Derived from the connectome (real):**

| | |
|---|---|
| Network connectivity | Real synaptic weights from `male-cns:v1.0`. |
| Excitatory / inhibitory sign | From `consensusNt`. Glutamate and histamine are treated as **inhibitory**, which is correct for flies (GluCl/HisCl) and opposite to the vertebrate intuition. |
| Descending / motor identity | `superclass in (descending_neuron, vnc_motor, cb_motor)`. There is no `class == 'descending neuron'` in this dataset. |
| Per-leg motor pools | `subclass` `fl`/`ml`/`hl` × `somaSide`, giving six populations that map onto the six legs. |
| Population sizes | 1,314 DNs; 708 + 107 MNs; 381 leg MNs. Queried live, not recalled. |

**Modelling approximations (not real):**

| | |
|---|---|
| **Fluid as density/viscosity** | MuJoCo's medium model, not CFD. Quadratic drag on body-sized shapes; no free surface, no wake, no wetting. "Submerged" means "immersed in a dense medium". |
| **Buoyancy** | MuJoCo's fluid model provides drag and lift but **no buoyancy** — verified: a sphere with exactly the medium's density still sinks. Buoyancy is therefore folded into an effective gravity in the preset (−300 rather than −9810), which is a stand-in, not a force balance. |
| **Swim arousal** | Swimming uses an elevated `tonic_drive` (1.40 against the default 1.05). Without it the mode settles into a self-sustaining torpor: with no ground contact the only limb load is fluid drag, so feeble strokes generate little sensory drive, which produces feebler strokes. A submerged insect struggling is real; the number is a placeholder. |
| **Swimming** | Thrust is pure drag asymmetry: fast power stroke with the legs spread, slow recovery with them folded. Measured in open water — the asymmetric stroke moves the fly, a symmetric one gives ~0.005 mm/s, i.e. nothing. At 0.6 mm/s it is ~18x slower than walking, which is not a tuning failure: a millimetre-scale body in water sits at Reynolds ~1–10, where viscosity dominates and rowing is a poor way to travel. Real adult *Drosophila* are bad swimmers. Enabling MuJoCo's per-geom ellipsoid fluid model on the legs was tried and made it *worse* (0.17 vs 0.67 mm/s), since it disables the inertia-box model for those bodies. |
| **Q10 = 2.3** | A **placeholder**, not a sourced constant. Drosophila are ectotherms and neural kinetics do scale with temperature, but the specific coefficient here is uncalibrated and depends on which process you mean. Fit it before using it. |
| **Muscle Q10** | The position actuators stand in for muscle and their gain is scaled by the *same* Q10 as the membrane, inverted (warm muscle is faster; a warm membrane time constant is shorter). Muscle and membrane need not share a coefficient — this reuse is a placeholder. The gain is clamped to [4, 40]: below ~4 the fly cannot hold itself up, above ~40 the stiffness buys nothing. |
| **`tau_clamp_ms`** | An integrator guard rail, not biology. |
| **Locomotion is CPG-generated** | The gait comes from six coupled oscillators. The connectome steers; it does not walk. |
| **Rate → drive mapping** | `tanh(rate / REFERENCE_RATE_HZ)` with hand-picked reference rates. A heuristic. |
| **Tonic drive (1.05)** | With `--neuron-subset motor` everything upstream of the DNs is missing, so nothing would reach threshold. A constant background current stands in for the absent network. |
| **Sensory encoding** | Contact force is injected into each leg's *motor* pool as a proprioceptive stand-in; a scalar air-motion term stands in for Johnston's organ. Real `vnc_sensory` neurons are used when the loaded subset contains them. Anatomically crude either way. |
| **Sensory gains per preset** | Reasoned, not measured. Humidity and immersion plausibly change mechanosensory and antennal input; the numbers are invented. |
| **Humidity → tarsal grip** | Insect tarsal adhesion genuinely is humidity-dependent, but the direction is regime-dependent and the literature is mixed: moderate humidity can *increase* attachment through capillary bridges at the pad, while a condensed film on the surface reduces it. This preset takes the wet-film case. The mechanism is real; the number (80 against 200) is a placeholder. |
| **Gait joint amplitudes** | Chosen by sweeping speed *and* postural stability together across coxa sweep, tibia sweep, lift, duty factor, stride frequency, actuator gain and adhesion. Reaches ~10 mm/s, the bottom of a real fly's ~10-20 mm/s, but the coxa excursion (2.6 rad ≈ 149°) is far beyond anything physiological. It moves the model convincingly; it is not measured Drosophila kinematics. |
| **Leg adhesion gain** | 200, against MuJoCo's default of 1.0. Without it the foot slips through stance and a stride delivers a fraction of the travel its geometry implies — this single parameter was worth about 3x in speed. Above ~400 the foot sticks hard enough to pull the fly off a straight line. |
| **Postural margin** | The model fly is stable over a narrow wind band and cannot right itself once over, so `windy` gusts to ~1.9 m/s and no further: 2.0 m/s capsizes it permanently and 2.5 m/s carries it away. Per-leg drive is limited to ±5%, which is what keeps the path straight. |
| **Gust model** | Two incommensurate sines per axis, so gusts do not look metronomic. It is not turbulence — there is no spatial structure, no eddies, and every part of the body sees the same wind at the same instant. |
| **Unknown neurotransmitters** | ~12k neurons have `unclear`/null `consensusNt` and are treated as excitatory, following the base rate. |
| **`humid_air` physics** | Humid air is very slightly *less* dense than dry air. The preset says so rather than inventing drag; its real effect is on the sensory gains. |
| **Odour as a Gaussian field** | Sources are static 2-D Gaussians summed in `body/world.py`. No advection, no turbulence, no filaments, and wind does not move them — the `windy` preset lowers `olfactory_gain` instead, which says "the cue is unreliable here" rather than pretending the model still holds. Plume widths are set by what the fly can navigate, not by chemistry. |
| **Olfaction injected at the DNs** | Concentration at the two antennae is injected straight into the left and right descending populations. That skips the entire antennal lobe / lateral horn / mushroom body pathway and pretends the descending command already carries the olfactory decision. It is the right *target* — olfactory receptor neurons are in the antennae, nowhere near `vnc_sensory` — and a huge shortcut in *depth*. |
| **Bilateral gain** | The antennae are 0.28 mm apart, so the left/right difference is ~1% of the concentration itself. A real fly closes that gap in *time* — casting its head and body and comparing successive samples — which a body with no neck joint cannot do. `ODOUR_BILATERAL_GAIN` stands in for the missing temporal comparison and is deliberately large; it was swept against how close the fly actually gets to a source, and the sweep is in the source. |
| **Receptor adaptation** | One leaky integrator per antenna. Real ORN adaptation is multi-timescale. |
| **No spontaneous turning** | Real flies make saccadic turns driven by the central complex, which is absent from the `motor` subset. Nothing here stands in for it — a stand-in was built and measured worse than nothing (see below), so what course changes the fly makes come from the odour loop and from the connectome's own left/right bias. |
| **The worktop scatter** | 300 crumbs and spills generated from a fixed seed out to 620 mm, because the fly nets ~9 mm/s and a hand-placed vignette runs out in half a minute. Their positions carry no meaning. |
| **Which props are solid** | Chosen by what the gait can survive, not by what a kitchen is like: anything the fly could wedge against permanently is scenery it passes through. See "The world has to be walkable". |
| **The sky** | Decorative. It emits no light, casts no shadow and has no effect on the simulation; the fly has vision hardware in FlyGym that this project does not use. |
| **Flight forces** | Lumped into a velocity controller applied as a wrench on the thorax. The wings are animated and generate none of it — measured at 0.072 body weights at best. See "Flight". |
| **Attitude and yaw control** | A PD loop standing in for the halteres. Real haltere feedback is a campaniform reflex arc through specific neurons, none of which are in the loaded subset. |
| **Airborne arousal** | Tonic drive is raised to 1.5 while flying. Same problem `submerged_water` documents: with no ground contact the leg pools starve of sensory input and the decoded drive decays to nothing — measured, the fly froze in mid-air to within 0.1 mm for 16 s. |
| **Obstacle avoidance in flight** | A latched contact reflex. Note this is the *opposite* conclusion to walking, where the same idea measured worse than nothing: flying, yaw comes from the attitude controller and works whatever the fly is touching. |
| **Knowing where the walls are** | The flight controller is told the room's bounds and turns back at the edge. A real fly does this with optic flow and the looming response; this model has eyes it does not use. Without it the fly worked itself into a corner and spent 25 s of a 50 s run climbing it. |
| **Takeoff and landing** | A behavioural rule on top of the olfactory loop, not a connectome mechanism: take off when a smell is exhausted, commit to a landing when a new one appears. Real takeoff is a giant-fibre escape or a voluntary sequence; neither is modelled. |

---

## CLI

```bash
python3 run.py --list-envs
python3 run.py --env submerged_water --steps 50000 --gpu 0 --render off
python3 run.py --world kitchen --env temperate --steps 40000
python3 run.py --compare-envs dry_land submerged_water windy --steps 20000
python3 run.py --connectome synthetic --neuron-subset motor --steps 1000
```

`--world {flat,kitchen,room}` picks the scenery, independently of `--env`. It
defaults to `flat` so the comparison numbers below stay reproducible. Only
`room` has air space in it, and only there does the fly get wings.

`--compare-envs` saves one brain snapshot and reloads it for every preset, so
differences between environments are not confounded by differences between
networks. It reports locomotion kinematics: mean/peak speed, net displacement,
path straightness, gait regularity and joint excursion.

Note that **gait regularity is a phase-locking measure** and, once the CPG has
locked, is largely drive-independent — it reveals gait *breakdown*, which does
not occur at these fluid values. Speed and displacement are the sensitive
metrics.

---

## Browser UI

`docker compose up web`, then <http://localhost:8080>. Live MuJoCo view over a
WebSocket (with an MJPEG fallback at `/stream.mjpg`), buttons for every preset,
pause/resume/reset, a DN-override slider, and a metrics readout.

It serves the **kitchen** world by default; set `FLY_WORLD=flat` for bare
ground. World choice is start-up only — swapping worlds means recompiling the
MuJoCo model and rebuilding the GL context, which cannot be done from the
control socket's thread. Presets still switch live.

**Real time.** The UI reports its own pace in the header (`19 fps · 0.85x real
time`) — the figure it measured, not a claim. It gets there by running the controller at 500 Hz against physics at
10 kHz — see [Real time](#real-time) — not by cutting corners in the physics.

**Brain activity.** The control socket also pushes a spike raster and the
population rates the decoder reads: the two DN pools that set `turn`, and the
six leg motor pools that set `forward`. Each raster cell is a spike *count* over
a 5 ms bin, not an instantaneous sample — a neuron at 100 Hz fires in about 2%
of the 0.2 ms brain steps, so sampling once per bin would show an empty screen.
Rows are drawn from those eight populations rather than at random, so you can
watch DN_L and DN_R separate when the fly turns toward a smell.

The DN override biases the decoded drive directly. It is a **debug lever**, not
a biological mechanism — it exists so the demo does not depend on waiting for
emergent behaviour.

The `web` service gets `/data/connectome` **read-only** and is never given
`NEUPRINT_TOKEN`. Read-only sim control.

---

## Requirements

CPU is the default and is enough for the `motor` subset.

| | |
|---|---|
| CPU | ~1.5 GB RAM for `motor`; physics runs ~2,500–7,000 steps/s single-threaded |
| GPU *(optional)* | CUDA 12.x needs host driver **≥ 525.60.13**; CUDA 13.x needs **≥ 580** |
| VRAM | `motor` ~0.5 GB, `vnc` ~2 GB, `all` ~8–12 GB (estimated from edge counts, **not measured**) |

Physics is CPU-only MuJoCo either way; the GPU only accelerates the LIF network,
so it pays off for `--neuron-subset vnc` and `all`, not for the default.

### Building behind a proxy

If `apt` and public PyPI are blocked but Docker Hub and an internal mirror are
reachable, use `Dockerfile.offline`. It bases on FlyGym's official image (which
already ships libEGL, ffmpeg, flygym and mujoco) so no `apt` is needed:

```bash
# .env next to docker-compose.yml
DOCKERFILE=Dockerfile.offline
PIP_INDEX_URL=https://your-mirror/artifactory/api/pypi/pypi/simple
```

That base image is `linux/amd64` only. It is used purely for the system
libraries it already carries (libEGL, ffmpeg); its own Python environment is
ignored and a clean venv is built on top.

### TLS behind an inspecting proxy

If your proxy re-signs TLS, Python will not trust it even where `curl` does, and
neuPrint fetches fail with `CERTIFICATE_VERIFY_FAILED`. Point Python at a bundle
that includes the proxy CA:

```bash
security find-certificate -a -p \
  /System/Library/Keychains/SystemRootCertificates.keychain > ca.pem   # macOS
export REQUESTS_CA_BUNDLE=$PWD/ca.pem
```

`--connectome synthetic` needs no network at all. The error message points this
out when it happens.

---

## Tests

```bash
./run.sh test                                          # native
docker compose run --rm sim python3 -m pytest tests/ -q  # in the container
```

| file | covers |
|---|---|
| `test_env_loader.py` | preset schema, the SI-units trap, the no-op-wind trap, Q10 direction and clamping, and that presets actually reach `mjOption` |
| `test_brain.py` | role counts vs the real dataset, fly-specific E/I signs, subsetting, refractory limits, checkpoint replay |
| `test_bridge.py` | six non-empty leg pools, bounded drive, sensory gain scaling |
| `test_cpg.py` | tripods lock antiphase, turning asymmetry, stop freezes the gait |
| `test_smoke.py` | 100 steps end to end; presets reach **both** physics and neurons; mid-run switching |
| `test_regression_env.py` | dry vs submerged changes joint kinematics and slows the fly |
| `test_world.py` | the kitchen keeps its ground-contact sensors, the sky animates and is reproducible, a smell on the right turns the fly right, receptors habituate, and the fly ends up nearer a source only when it can smell |
| `test_flight.py` | the wingbeat strokes *and* feathers, the controller holds a hover and caps its manoeuvre force, attitude damping does not swamp the turn command, wings exist only in a flyable world, and landing clears the applied wrench |
| `test_realtime.py` | a decimated control rate keeps the gait, keeps the sensory loop closed, and reproduces the default model exactly at `control_every=1` |

---

## Verified vs not

Developed on Apple Silicon macOS behind a filtering corporate proxy. Being
explicit about what that means:

**Actually run and verified**

- All 73 tests pass, from a clean checkout via `./run.sh test`.
- **The real connectome runs.** `male-cns:v1.0` fetched live from neuPrint:
  2,129 neurons and **104,411 real synaptic edges**.
- **The fly stays upright and walks in all six presets** for 30,000 steps
  (3 s of simulated time) — verified on body roll/pitch, not just height.
- `--compare-envs` on real connectome data, 30,000 steps each:

  | metric | cold 10C | dry_land | temperate | humid_air | hot 35C | windy | submerged |
  |---|---|---|---|---|---|---|---|
  | net speed mm/s | **1.66** | 7.68 | **10.75** | 5.20 | 5.83 | 6.01 | **0.86** |
  | leg speed mm/s | **4.7** | 21.4 | 22.5 | 20.7 | **29.0** | 24.6 | 3.1 |
  | feet on the ground | **4.05** | 2.8 | 2.73 | 2.8 | **2.23** | 2.7 | **0.2** |
  | slip ratio | 0.65 | 0.63 | **0.52** | 0.75 | **0.80** | 0.72 | n/a |
  | straightness | 0.882 | 0.817 | **0.989** | 0.689 | **0.435** | **0.489** | **0.482** |

  Two of the axes are non-monotonic, so `temperate` is the best case on both:

  * **Temperature peaks at 25 C and fails in OPPOSITE ways at the two ends.**
    **Cold (10 C)** is near chill coma - *Drosophila* CTmin is roughly 4-8 C.
    Both Q10 channels bite at once: membrane time constants stretch to ~70 ms
    and muscle gain falls to 5.7, so the fly cannot hold itself up. It sags and
    drags **4.05 feet** along the ground rather than running a three-point
    tripod, with its legs barely moving (4.7 mm/s). **Hot (35 C)** is the
    opposite failure: the nerve outruns the muscle. Leg pools fire at ~123 Hz,
    the network commands a ~29 Hz stride the actuators cannot track, and the fly
    thrashes - the highest leg speed of any preset (29.0), the FEWEST feet down
    (2.23), 80% of the motion wasted and the path collapsing to 0.435.
    Sluggish at one end, frantic at the other; net speed alone would confuse
    them, which is why leg speed and feet-down are in the table.
  * **Humidity peaks at ~60% RH** for an unrelated reason: tarsal pads need some
    moisture to form the capillary bridges that grip, so dry air weakens them,
    while a condensed film at saturation makes them slip.
  * **Wind** gusts rather than blowing steadily - a steady wind can only bias
    the path or delete it (2.0 m/s capsizes the fly, 2.5 carries it away).
  * **Water** is a different mode of locomotion: suspended, rowing all six legs.

  Three numbers, three meanings. **Net speed** is displacement over elapsed time
  and is the honest walking speed. **Mean speed** averages instantaneous
  |velocity| and is inflated by per-stride body sway. **Straightness** is
  measured on a stride-independent decimation of the path: sampled faster than
  one stride it charges the fly for its own sway and reports ~0.43 for a
  trajectory that is actually near-straight. Path length is not a
  sampling-rate-free quantity, so any "mm/s along the path" figure should be
  distrusted - including one quoted earlier in this project's history, before
  the interval was pinned.

- The web UI: WebSocket frame streaming, preset switching, DN override,
  pause/resume/reset, **and the brain-activity raster** all confirmed against a
  running server. The raster was checked against the reported rates rather than
  by eye: the two DN bands come out 17–20% lit at 33–36 Hz and the six leg bands
  27–32% lit at 52–55 Hz, in the right order.
- **The kitchen world**, on the synthetic connectome: the fly tracks in on a
  smell, habituates, leaves and finds another, over 150k-step runs on two seeds.
  The odour gain, the plume width and the spontaneous-turn experiment were all
  swept in-situ; the numbers are in the source next to the constants they set.
- **`--world flat` is unchanged by the kitchen.** 12,000-step Sessions in
  `dry_land`, `windy` and `cold` gave bit-identical metrics before and after it
  landed. The later SciPy matmul reorders a float32 sum and so diverges
  chaotically — same model, ~1e-7 relative on `dry_land`, 0.1% on `windy` after
  12,000 steps.
- **Real time for the simulation**: 1.41x walking and 1.42x flying in the room
  against 0.37x for the documented model — a 3.8x speed-up, reached by
  decoupling the control rate from the physics rate after measuring that the
  timestep cannot be raised (the fly falls over) and that `mj_step` was only a
  fifth of the loop. Measured in CPU time, because the development machine was
  loaded to 15+ and wall-clock timing there had already produced one confidently
  wrong conclusion. Metal was measured and rejected for the default subset. See
  [Real time](#real-time).
- `Dockerfile.offline` builds and its tests pass inside the container.
- `Dockerfile` (the non-offline one) builds and runs on an unrestricted
  network: `docker build .` completes, and the built image fetched a real
  connectome from neuPrint during the data-fetch stage (2,129 neurons, 104,411
  edges) and ran `run.py --list-envs` correctly. Not re-run under `pytest`
  inside this container specifically.
- **`docker compose build`, `docker compose up web` and
  `docker compose run --rm sim` all work**, on linux/arm64 (Apple Silicon,
  OrbStack) — after the fix described in
  [A Docker trap](#a-docker-trap-for-the-record). Note that the plain
  `docker build .` above was never affected by that bug: it only appears through
  compose, which passes an explicitly empty `TORCH_INDEX`. The kitchen renders
  headless through EGL in the container and the browser UI serves from it.
  A container has no GPU, so it rasterises in software and the live view is much
  slower there than natively; use Docker for batch runs, where `RENDER=off` is
  the default anyway.

### A failure worth recording

An earlier version of the gait defined stance as the wrong half of the step
cycle, so each leg extended into the ground with adhesion off. That levers the
body upward: the fly reared monotonically (pitch climbing past +50 deg) and
capsized onto its back within a few thousand steps, then flailed there
indefinitely.

It passed every test at the time. Two things hid it:

* `fell_over` only checked thorax height, and an upended fly's thorax sits well
  above the floor. It now checks roll and pitch as well.
* The open-loop gait tests drove the CPG at `forward = 1.0`, while the brain
  actually commands about 0.2. The rearing only appears at low stride
  frequency, so a fast test never saw it.

`test_regression_env.py::test_the_fly_stays_upright` exists to stop this
recurring, and the windy preset is documented with the crosswind at which the
model capsizes.

### Four more, from building the kitchen

Two of these are the same lesson twice, which is why both are here: a mechanism
that is obviously necessary can be measured against not having it, and lose.

**A reflex that made things worse.** The fly kept wedging against props, so the
obvious fix was an obstacle-avoidance reflex: contact on the left drives DN_R,
the fly turns away. Measured against a 3.6 mm cube on its path, over 60k steps:

| escape turn | steps in contact | got past it |
|---|---|---|
| none | 13,214 | yes, at step 30,306, and walked 33 mm clear |
| contralateral, 0.35 | 46,372 | no |
| contralateral, 0.80 | 35,391 | no |
| ipsilateral (wall-follow), 0.35 | 5,283 | no |
| ipsilateral (wall-follow), 0.80 | 26,297 | no |

Every variant was worse than doing nothing. Turning while pressed against a
face only re-aims the fly into it; left alone, the same physics slides it along
the face and off the end. The reflex was deleted and the world was changed
instead — obstacles the fly cannot slide past are simply not solid. The lesson
is narrow and worth keeping: an intervention that is obviously right can be
measured, and this one lost to the empty control.

**A spontaneous-turn drive that made things worse.** Chemotaxis can only pull
the fly towards a plume it happens to pass, and with `--neuron-subset motor`
nothing in the loaded network can produce a course change on its own — the
central complex, which is where a fly's spontaneous turns come from, is missing
entirely. So an Ornstein–Uhlenbeck bias between the DN pools was added to stand
in for it, tuned against a measured current-to-turn curve so that it sat in the
same band as the olfactory drive. Over 150k steps in the kitchen, two seeds:

| | time within 15 mm of a source | straightness | furthest from spawn |
|---|---|---|---|
| spontaneous turning on | 4.7% / 8.1% | 0.97 / 0.96 | 144 / 141 mm |
| off | **28.8% / 25.0%** | **0.77 / 0.77** | **116 / 116 mm** |

Worse on every measure and in the same direction both times. The random bias
kicks the fly out of the loops the odour gradient puts it into, so it ends up
*straighter* — the opposite of the intended effect. Deleted.

**Two frames that had to match.** Getting the fly to fly took three bugs that
all looked like tuning problems and were not:

* MuJoCo reports a free joint's angular velocity in the **body** frame while
  `xfrc_applied` takes a torque in the **world** frame. Level, the two coincide,
  so a hover looked perfect; the moment the fly tilted the attitude damping
  became positive feedback and flipped it onto its back.
* `xfrc_applied` acts at the *thorax body's* centre of mass, but the fly's mass
  is spread over legs, head, abdomen and wings and its real centre of mass is
  ~0.5 mm behind that. One body weight of pure forward force spun the fly to
  9.3 rad/s of pitch in 0.3 s; in flight it showed up as a 13 rev/s spin the
  instant any thrust was commanded.
* The attitude loop was damped at a quarter of critical for the fly's
  rotational inertia. That is invisible starting from equilibrium and tumbles on
  the first disturbance.

**A sky that was clipped away.** FlyGym ships `zfar = 250` (× the unit model
extent, so 250 mm). The first version of the kitchen had a mug, a toaster and a
backsplash on the horizon that never appeared in a single frame — no error, no
warning, just nothing there.

**Plumes that were too wide to follow.** Fifteen odour sources at σ 15–22 mm
with ~50 mm spacing overlap into a smooth field with no gradient left in it: the
fly walked 150k steps at straightness 0.98 and never came within 12 mm of
anything. Halving σ separated the plumes and it started tracking. More signal
made the problem worse, which is not the direction intuition points.

**Written but NOT executed**

- **Everything CUDA.** No NVIDIA GPU was available; `nvidia/cuda` images are
  amd64-only and the dev host is arm64. The CUDA build args, `gpus: all` and the
  `--gpu` code path are unverified. Treat the VRAM figures as estimates.
- **The Feather path.** `storage.googleapis.com` is blocked on the development
  network, so the bulk files were never downloaded and their column names were
  never confirmed. Run the `--schema` command above before trusting it.
- **The compose path on linux/amd64.** `docker compose build` is verified on
  arm64 only; the amd64 path is the same recipe with the same wheels and should
  behave, but it has not been run since the fix.

---

## Licence and attribution

Connectome data: Janelia FlyEM male CNS, CC-BY 4.0
(<https://male-cns.janelia.org/>). The body model comes from FlyGym /
NeuroMechFly (<https://github.com/NeLy-EPFL/flygym>); please cite them, not this
repository, for anything anatomical.

# Connectome-driven fly simulation

A biomechanical *Drosophila* body (FlyGym / NeuroMechFly v2 on MuJoCo) driven by
a sparse spiking network whose **connectivity** comes from the Janelia male CNS
connectome (`male-cns:v1.0`), runnable across six environment presets that
perturb both the physics and the neural dynamics.

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
./run.sh compare dry_land submerged_water windy
./run.sh web                          # http://localhost:8000
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

Browser UI on <http://localhost:8080>:

```bash
docker compose up web
```

---

## What it does

```
CONNECTOME  (neuPrint male-cns:v1.0)
  -> sparse LIF network (PyTorch, CPU or CUDA)
  -> descending neurons + VNC motor neurons as the output layer
       |
BRIDGE   decode MN/DN population rates -> descending drive
         encode contact + air motion -> injected current
       |
BODY     tripod CPG -> 42 leg joint targets -> MuJoCo
ENV      preset YAML -> MuJoCo fluid options AND LIF time constants
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
| `dry_land` | still dry air | 25 °C | baseline |
| `humid_air` | near-saturated air | 25 °C | sensory gains only (see note below) |
| `submerged_water` | water | 20 °C | ~1000× density, `implicitfast` integrator |
| `hot` | dry air | 35 °C | faster neural kinetics (Q10) |
| `cold` | dry air | 15 °C | slower neural kinetics (Q10) |
| `windy` | dry air + 1.5 m/s | 25 °C | strong lateral wind |

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
| `wind` | mm/s | `1500` = 1.5 m/s | — | ×1e3 |
| `gravity` | mm/s² | `-9810` | | ×1e3 |

Every preset declares `units: mm_g_s` and the loader rejects anything else.
`tests/test_env_loader.py` asserts water's density is ~1e-3 and not ~1000.

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
| **Swimming** | Thrust is pure drag asymmetry: fast power stroke with the legs spread, slow recovery with them folded. Measured in open water — the asymmetric stroke moves the fly, a symmetric one gives ~0.005 mm/s, i.e. nothing. At 0.6 mm/s it is ~18x slower than walking, which is not a tuning failure: a millimetre-scale body in water sits at Reynolds ~1–10, where viscosity dominates and rowing is a poor way to travel. Real adult *Drosophila* are bad swimmers. Enabling MuJoCo's per-geom ellipsoid fluid model on the legs was tried and made it *worse* (0.17 vs 0.67 mm/s), since it disables the inertia-box model for those bodies. |
| **Q10 = 2.3** | A **placeholder**, not a sourced constant. Drosophila are ectotherms and neural kinetics do scale with temperature, but the specific coefficient here is uncalibrated and depends on which process you mean. Fit it before using it. |
| **`tau_clamp_ms`** | An integrator guard rail, not biology. |
| **Locomotion is CPG-generated** | The gait comes from six coupled oscillators. The connectome steers; it does not walk. |
| **Rate → drive mapping** | `tanh(rate / REFERENCE_RATE_HZ)` with hand-picked reference rates. A heuristic. |
| **Tonic drive (1.05)** | With `--neuron-subset motor` everything upstream of the DNs is missing, so nothing would reach threshold. A constant background current stands in for the absent network. |
| **Sensory encoding** | Contact force is injected into each leg's *motor* pool as a proprioceptive stand-in; a scalar air-motion term stands in for Johnston's organ. Real `vnc_sensory` neurons are used when the loaded subset contains them. Anatomically crude either way. |
| **Sensory gains per preset** | Reasoned, not measured. Humidity and immersion plausibly change mechanosensory and antennal input; the numbers are invented. |
| **Gait joint amplitudes** | Chosen by sweeping speed *and* postural stability together across coxa sweep, tibia sweep, lift, duty factor, stride frequency, actuator gain and adhesion. Reaches ~10 mm/s, the bottom of a real fly's ~10-20 mm/s, but the coxa excursion (2.6 rad ≈ 149°) is far beyond anything physiological. It moves the model convincingly; it is not measured Drosophila kinematics. |
| **Leg adhesion gain** | 200, against MuJoCo's default of 1.0. Without it the foot slips through stance and a stride delivers a fraction of the travel its geometry implies — this single parameter was worth about 3x in speed. Above ~400 the foot sticks hard enough to pull the fly off a straight line. |
| **Postural margin** | `windy` runs at 1.5 m/s, which visibly shoves the fly without knocking it over. At 2.0 m/s it is lifted off the floor and tumbles away — a cliff, not a gradient. Per-leg drive is limited to ±5%, which is what keeps the path straight. |
| **Unknown neurotransmitters** | ~12k neurons have `unclear`/null `consensusNt` and are treated as excitatory, following the base rate. |
| **`humid_air` physics** | Humid air is very slightly *less* dense than dry air. The preset says so rather than inventing drag; its real effect is on the sensory gains. |

---

## CLI

```bash
python3 run.py --list-envs
python3 run.py --env submerged_water --steps 50000 --gpu 0 --render off
python3 run.py --compare-envs dry_land submerged_water windy --steps 20000
python3 run.py --connectome synthetic --neuron-subset motor --steps 1000
```

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

---

## Verified vs not

Developed on Apple Silicon macOS behind a filtering corporate proxy. Being
explicit about what that means:

**Actually run and verified**

- All 46 tests pass, from a clean checkout via `./run.sh test`.
- **The real connectome runs.** `male-cns:v1.0` fetched live from neuPrint:
  2,129 neurons and **104,411 real synaptic edges**.
- **The fly stays upright and walks in all six presets** for 30,000 steps
  (3 s of simulated time) — verified on body roll/pitch, not just height.
- `--compare-envs` on real connectome data, 30,000 steps each:

  | metric | cold 15C | dry_land 25C | hot 35C | windy | submerged |
  |---|---|---|---|---|---|
  | net speed mm/s | 5.38 | 10.77 | **11.47** | 6.95 | **0.61** |
  | peak speed mm/s | 51.4 | 45.0 | 47.4 | **78.4** | 17.7 |
  | straightness | 0.962 | 0.990 | 0.999 | **0.703** | **0.345** |
  | feet on the ground | 2.7 | 2.7 | 2.7 | 2.7 | **0.2** |
  | fell over | no | no | no | no | n/a (swimming) |

  Every preset does something distinct, for a different reason:

  * **Temperature** spans ~2x in walking speed. The causal chain is the intended
    one: Q10 shortens the membrane time constants, leg motor pools fire faster
    (35 / 70 / 123 Hz at 15 / 25 / 35 C), the descending drive rises and the CPG
    steps quicker. Ectotherms are slower when cold, so the direction is right.
    Hot is limited by the BODY: above ~22 Hz stride the position actuators stop
    tracking, which is what `MAX_FORWARD_DRIVE` is pinned to.
  * **Wind** at 1.5 m/s shoves the fly off its heading (straightness 0.70) while
    gusting it to 78 mm/s peak, far faster than it can walk.
  * **Water** is a different mode of locomotion, not a slow walk: the fly is
    suspended (0.2 feet touching, against 2.7 on land), rowing all six legs in
    synchrony. It manages 0.6 mm/s, and that is the honest answer - see below.

  Three numbers, three meanings. **Net speed** is displacement over elapsed time
  and is the honest walking speed. **Mean speed** averages instantaneous
  |velocity| and is inflated by per-stride body sway. **Straightness** is
  measured on a stride-independent decimation of the path: sampled faster than
  one stride it charges the fly for its own sway and reports ~0.43 for a
  trajectory that is actually near-straight. Path length is not a
  sampling-rate-free quantity, so any "mm/s along the path" figure should be
  distrusted - including one quoted earlier in this project's history, before
  the interval was pinned.

- The web UI: WebSocket frame streaming, preset switching, DN override, and
  pause/resume/reset all confirmed against a running server.
- `Dockerfile.offline` builds and its tests pass inside the container.

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

**Written but NOT executed**

- **Everything CUDA.** No NVIDIA GPU was available; `nvidia/cuda` images are
  amd64-only and the dev host is arm64. The CUDA build args, `gpus: all` and the
  `--gpu` code path are unverified. Treat the VRAM figures as estimates.
- **The Feather path.** `storage.googleapis.com` is blocked on the development
  network, so the bulk files were never downloaded and their column names were
  never confirmed. Run the `--schema` command above before trusting it.
- **`Dockerfile` (the non-offline one).** `apt` could not reach Debian mirrors
  from containers on this network, so the standard build was never completed
  here. It is the conventional recipe and should work on an unrestricted
  network, but it is unproven.

---

## Licence and attribution

Connectome data: Janelia FlyEM male CNS, CC-BY 4.0
(<https://male-cns.janelia.org/>). The body model comes from FlyGym /
NeuroMechFly (<https://github.com/NeLy-EPFL/flygym>); please cite them, not this
repository, for anything anatomical.

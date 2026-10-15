# UMA-CBridge

UMA-CBridge connects the **UMA machine-learning interatomic potential** to Gaussian, ORCA, AMBER, CREST, and i-PI. It lets these programs use UMA energies and gradients for geometry optimization, molecular dynamics, and other workflows supported by their external interfaces. UMA is developed by Meta's FAIR Chemistry team; this project provides the server and client wrappers that connect it to those programs.

**UMA-CBridge uses compiled C wrappers to substantially reduce interface and communication overhead.** Several existing open-source UMA interfaces use Python wrappers, including [ORCA_wrapper_UMA](https://github.com/leforesb/ORCA_wrapper_UMA) and [umas](https://github.com/npe1011/umas). UMA-CBridge uses small C executables for Gaussian, ORCA, AMBER, and CREST: they exchange binary numerical data with the resident UMA server without starting a Python interpreter or importing Python dependencies on each call. Measurements by the author show substantial time savings in these repeated interface calls, which are frequent during optimization, molecular dynamics, and conformational searches.

The server loads the UMA model once and keeps it in memory for repeated calculations, using FAIRChem's `omol` task. It supports queued energy/gradient batches, batched finite-difference Hessians, and optional MoLE merging and model compilation. All Python functionality is in **`UMA_server.py`**, with editable settings at the top; in i-PI mode, this script acts as the force client.

**The `orca` executables supplied for AMBER and CREST are fake ORCA adapters.** They imitate the ORCA interface so that AMBER and CREST accept them as the requested external program. They read the structures, send them to the UMA server, and return UMA energies and gradients in ORCA-compatible `.engrad` files and output messages. They perform no ORCA electronic-structure calculation and need no ORCA installation. Method and basis keywords in their inputs do not change the UMA model. The separate `wrappers/orca/orca_uma` client is used by **real ORCA** through ExtOpt.

UMA resources: [official website and documentation](https://fair-chem.github.io/), [FAIRChem source code](https://github.com/facebookresearch/fairchem), and [UMA model downloads](https://huggingface.co/facebook/UMA). When publishing results obtained with UMA, cite Brandon M. Wood et al., *UMA: A Family of Universal Models for Atoms*, arXiv:2506.23971 (2025), [paper](https://arxiv.org/abs/2506.23971), [DOI](https://doi.org/10.48550/arXiv.2506.23971).

## Files

```text
UMA_server.py                    Server and i-PI client; the only Python script
requirements.txt                 Reference Python dependency versions
Makefile                         Build all four C clients
wrappers/
  common/fastuma_protocol.h       Shared socket protocol
  gaussian/gaussian_uma.c         Gaussian External source
  gaussian/gaussian_uma           Gaussian External executable
  orca/orca_uma.c                 ORCA ExtOpt source
  orca/orca_uma                   ORCA ExtOpt executable
  amber/orca.c                   AMBER fake ORCA adapter source
  amber/orca                     AMBER fake ORCA executable
  crest/orca.c                   CREST fake ORCA adapter source
  crest/orca                     CREST fake ORCA executable
examples/                        Small inputs for each interface
LICENSE                          MIT license for this project
```

## Build the C wrappers

Set up your UMA environment by following the [official FAIRChem/UMA documentation](https://facebookresearch.github.io/fairchem/) and obtain a checkpoint from the [official UMA model repository](https://huggingface.co/facebook/UMA). Run `UMA_server.py` in that environment. This project supplies the interfaces and does not include model weights.

To compile the four C wrappers, use a C11 compiler (GCC or Clang) and `make` on Linux. From the project root, run:

```bash
make -B
```

The included executables were built for Linux x86-64. `make -B` rebuilds all four wrappers for your machine; `make clean` removes them. The C wrappers themselves require no Python environment.

## Configure and start

Edit the configuration block at the top of `UMA_server.py`:

```python
MODEL_PATH = "/absolute/path/models/uma-s-1p2.pt"  # Required: absolute path to your UMA checkpoint (.pt).
```

`MODEL_PATH` is the only required setting and starts empty. Fill it in before starting the server. The settings below it are `PORT`, `DEVICE`, default inference settings, MD/i-PI inference settings, queue batching limits, and the four Hessian parameters, in that order.

```bash
python UMA_server.py
```

To enable queue batching:

```bash
OMP_NUM_THREADS=4 python UMA_server.py --batch
```

**Batched inference makes fuller use of the GPU and can significantly improve throughput**, especially when individual structures are small. With `--batch`, the server evaluates multiple ready energy/gradient requests together, so concurrent client jobs can share a GPU efficiently. The supplied i-PI example batches 32 beads through the XML `batch_size` setting; Hessian calculations batch displaced structures separately.

Larger batches also consume more GPU memory. Choose batch sizes for your GPU's available memory and the number of atoms per structure. For wrapper requests, `RAW_BATCH_MAX_SIZE` limits the number of jobs and `RAW_BATCH_MAX_ATOMS` sets the total atom budget. On CUDA out-of-memory errors, the server splits the affected request batch in half and retries. Adjust i-PI batches in the XML and Hessian batches with the `HESSIAN_BATCH_*` settings at the top of the script.

The raw server listens on `127.0.0.1:50423`. Keep this terminal running while a client calculates. Ctrl+C stops the service.

| Command option | Behavior |
| --- | --- |
| none | Process requests sequentially; TF32, MoLE merging, and compilation default to off |
| `--batch` | Evaluate queued energy/gradient requests together |
| `--md` | Use the editable `MD_INFERENCE_*` settings for one fixed system |
| `--batch --md` | Queue batching with the MD settings |
| `--ipi` | Connect to an i-PI server using the MD settings |
| `--verbose` | Print per-request or per-batch timings |

**TF32 is a manual setting, off by default in every mode.** `--md` and `--ipi` select the values below; they do not set TF32 to `True` automatically:

```python
MD_INFERENCE_TF32 = False
MD_INFERENCE_MERGE_MOLE = True
MD_INFERENCE_COMPILE = True
```

To enable TF32 for `--md`/`--ipi`, set `MD_INFERENCE_TF32 = True` in the script. Ordinary modes keep TF32 off.

MoLE merging requires fixed element composition, charge, and spin. Restart the server for a different system, or leave merging off for mixed workloads. The first real request initializes the merged/compiled model and may take substantially longer. Changing batch shapes can trigger additional compilation.

`PORT` defaults to `50423`; `DEVICE` defaults to `"cuda"` and can be changed to `"cpu"`.

All C clients accept `FASTUMA_HOST` (numeric IPv4 address, default `127.0.0.1`) and `FASTUMA_PORT` (default `50423`). To use another local port, edit `PORT` at the top of `UMA_server.py` and set the client environment variable `FASTUMA_PORT` to match. The server binds only to loopback and has no authentication.

## Gaussian

Set Gaussian's `External` command to the absolute path of `wrappers/gaussian/gaussian_uma`. Edit that path in `examples/gaussian/water.gjf`, then run with your Gaussian installation:

```bash
cd examples/gaussian
g16 < water.gjf > water.log
```

The client supports energy, gradient, and Hessian requests. Hessians are central differences of UMA forces; dipole moments, polarizabilities, and dipole derivatives are written as zeros. This interface supplies potential-energy derivatives, not electronic properties or IR intensities.

## ORCA ExtOpt

Edit `ProgExt` in `examples/orca/water.inp`:

```text
! ExtOpt
%method
  ProgExt "/absolute/path/UMA-CBridge/wrappers/orca/orca_uma"
end
* xyzfile 0 1 water.xyz
```

Run **real ORCA** in the example directory:

```bash
cd examples/orca
/absolute/path/to/real/orca water.inp > water.out
```

ORCA creates its external-interface input and invokes the client. The client reads that interface file, not a normal ORCA `.inp`, and returns energy/gradient in `<xyz-basename>.engrad`. See the [ORCA ExtOpt manual](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html#external-optimizers).

## AMBER through the ORCA interface

The example follows the external-QM and ORCA-template interface in the [AMBER 24 manual, Sections 10.1.6 and 10.2.6.4](https://ambermd.org/doc12/Amber24.pdf). Run it with `sander` and place `orc_job.tpl` in the working directory.

`examples/amber/qmmm_nvt.in` is a periodic NVT continuation input: 300 K, Langevin thermostat (`gamma_ln=1.0` ps⁻¹), 0.5 fs time step, and 100,000 steps (50 ps). It expects a prepared topology and an equilibrated restart containing coordinates, velocities, and box information (`irest=1, ntx=5`). The QM selection is residue 1, neutral singlet; change `qmmask`, `qmcharge`, and `spin` for your system. Keep complete molecules in the QM region for this example. MM bonds involving hydrogen use SHAKE, while QM bonds remain flexible (`qmshake=0`).

The interface settings are:

```text
&cntrl
  ...
  ifqnt = 1,
/
&qmmm
  qmmask = ':1',
  qmcharge = 0, spin = 1,
  qm_theory = 'EXTERN',
  qmmm_int = 5,
  qmshake = 0,
  qm_ewald = 0, qm_pme = 0,
/
&orc
  use_template = 1,
  num_threads = 1,
  dipole = 0,
/
```

`qmmm_int=5` selects mechanical embedding: AMBER handles QM–MM electrostatics with topology charges and the classical nonbonded interactions, while UMA evaluates the QM subsystem. The adapter does not calculate polarization by MM point charges or their gradients. QM electrostatic Ewald/PME options are disabled; the periodic MM calculation remains AMBER's responsibility.

The supplied `orc_job.tpl` contains a method/basis placeholder (`! HF STO-3G`). The adapter ignores that line and uses the server's UMA checkpoint. **Do not put `EnGrad`, `%pointcharges`, `*xyzfile`, coordinates, charge, or multiplicity in this template.** Sander adds the calculation request and geometry to `orc_job.inp` and writes `inpfile.xyz`; charge and multiplicity come from `&qmmm`. These generated files are not distributed as example inputs.

From the project root, start `python UMA_server.py --md` in its own terminal. In the AMBER environment, select the adapter and launch the simulation:

```bash
export PATH="/absolute/path/UMA-CBridge/wrappers/amber:$PATH"
command -v orca
cd /absolute/path/UMA-CBridge/examples/amber
sander -O -i qmmm_nvt.in -o qmmm_nvt.out \
  -p /absolute/path/system.prmtop -c /absolute/path/equilibrated.rst7 \
  -r qmmm_nvt.rst7 -x qmmm_nvt.nc -inf qmmm_nvt.mdinfo
```

`command -v orca` should resolve to `wrappers/amber/orca`. Each simulation needs its own working directory and copy of `orc_job.tpl`. The adapter returns `orc_job.engrad`; its ORCA-style termination line reports a completed UMA calculation. Keep `dipole=0`, since this adapter supplies energy and gradients only.

## CREST through the ORCA interface

Edit the two absolute paths in `examples/crest/water.toml`, then run:

```bash
cd examples/crest
crest water.toml
```

The example uses `method="orca"`, `orca_cmd` pointing to `wrappers/crest/orca`, and the minimal `orca_template.inp`. The adapter accepts inline `* xyz` blocks and `* xyzfile` inputs. It writes the `.engrad` beside the input and replaces the output atomically.

The example performs geometry optimization (`runtype="ancopt"`). For conformer sampling, select `runtype="imtd-gc"` and choose the sampling settings for your system. Keep `threads=1` with CREST 3.0.2: its ORCA route can reuse work-file paths across workers. Independent CREST runs can use separate working directories and share a `--batch` server. See the [CREST ORCA interface documentation](https://crest-lab.github.io/crest-docs/page/examples/orca.html) for the options supported by your version.

Optional adapter diagnostics: `CREST_ORCA_LOG=/absolute/path/calls.log` records calls; `CREST_ORCA_FSYNC=1` syncs each gradient file before replacement. These do not change model settings.

## i-PI

In this mode **i-PI is the server and UMA-CBridge is the force client**. i-PI controls integration, masses, thermostat, and ring-polymer springs. UMA-CBridge returns the physical potential, forces, and virial for each independent structure.

For i-PI installation, follow the [official i-PI guide](https://docs.ipi-code.org/getting-started.html).

Terminal 1, in your i-PI environment, starting from the project root:

```bash
cd examples/ipi
i-pi methanol.xml
```

Terminal 2, in your UMA environment, from the project root with `MODEL_PATH` configured:

```bash
OMP_NUM_THREADS=4 python UMA_server.py --ipi \
  --ipi-unix fastuma --ipi-structure examples/ipi/methanol.xyz \
  --ipi-charge 0 --ipi-multiplicity 1
```

Pass the structure for each job with `--ipi-structure`; it can be the same XYZ file used to initialize i-PI. The isolated-methanol example uses 32 beads at 300 K, a 0.25 fs time step, and a PILE-G thermostat with a 100 fs centroid coupling time and `pile_lambda=1.0`. It runs 100,000 steps (25 ps) in NVT with a fixed cell and no barostat. Allow the first 5 ps for equilibration, extending that interval if the observables have not settled; assess bead-count and sampling convergence for the intended observable. Properties are written every 10 steps, the centroid trajectory every 100 steps, and a restart every 1,000 steps. See the [i-PI thermostat settings](https://docs.ipi-code.org/input-tags.html) for PILE options. Compilation on the first request may take minutes; the XML allows 1800 seconds.

Keep the template and i-PI coordinates in identical atom order. The socket protocol does not transmit elements, masses, charge, spin, or PBC flags. Set charge and multiplicity on the client; masses belong in i-PI. Keep XML `<parameters>` empty. Use an active subsystem only when the template matches that ordered subsystem exactly.

The supplied i-PI example enables batching by default with `<batch_size>32</batch_size>` in the XML; no `--batch` flag is needed. i-PI controls the batch size; `--ipi-max-batch` is only a limit (default 32). The batch size must stay fixed during a run. Leave message consolidation enabled and avoid `matching=lock` for batched runs. Older i-PI versions can omit `batch_size` and use sequential requests. See the [i-PI socket documentation](https://docs.ipi-code.org/distributed.html).

UNIX address `fastuma` maps to `/tmp/ipi_fastuma`; an absolute socket path is accepted. For TCP, set the XML socket to `mode="inet"` with the matching address and use `--ipi-host HOST`. Omitting `<port>` in the XML and `--ipi-port` on the command line uses `65535` on both sides. `--ipi-connect-timeout` defaults to 120 seconds. `--ipi-timeout 0` leaves socket I/O waits unlimited.

For isolated molecules, keep XML `pbc="False"` and omit `--ipi-pbc`. The virial uses centered positions and forces. For full 3D periodic calculations, add `--ipi-pbc` and configure the XML accordingly; the virial uses `-V × stress`. The `omol` stress is derived from energy and was not fitted to stress labels, so pressure-dependent applications need system-specific validation.

## Units and scope

Raw clients send atomic numbers and float32 coordinates in Å. Responses contain float64 energy in Hartree, gradients in Hartree/Bohr, and, when requested, the lower Hessian triangle in Hartree/Bohr². The raw protocol retains the original conversion constants. i-PI uses float64 Bohr/Hartree units with ASE conversions. Gradients are the negative of forces.

The raw interfaces model isolated structures. External electric fields, point charges, electronic densities, and arbitrary ORCA/Gaussian method keywords are not inputs to UMA. Choose a checkpoint and task appropriate to your chemistry.

Copyright (c) 2026 Yuchen Zhou ([RowenaZireael](https://github.com/RowenaZireael)). The project code is licensed under [MIT](LICENSE). UMA checkpoints and third-party programs retain their own licenses.

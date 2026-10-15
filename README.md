# UMA-CBridge

[中文说明](README_zh.md)

UMA-CBridge connects Meta FAIR Chemistry's **UMA machine-learning potential** to Gaussian, ORCA, AMBER, CREST, and i-PI for geometry optimization, molecular dynamics, and conformational sampling.

**C wrappers and a resident UMA server reduce interface and communication overhead.** Existing interfaces such as [ORCA_wrapper_UMA](https://github.com/leforesb/ORCA_wrapper_UMA) and [umas](https://github.com/npe1011/umas) use Python wrappers. Here, compiled C clients exchange binary data with the server, saving repeated Python startup and import time. The author has measured substantial savings in repeated calls.

**AMBER and CREST use fake ORCA executables written in C.** They impersonate ORCA and return UMA energies and gradients as `.engrad` files. Gaussian and ORCA use their External and ExtOpt interfaces. 

UMA: [official documentation](https://facebookresearch.github.io/fairchem/) · [source](https://github.com/facebookresearch/fairchem) · [model weights](https://huggingface.co/facebook/UMA). Cite Brandon M. Wood et al., [*UMA: A Family of Universal Models for Atoms*](https://arxiv.org/abs/2506.23971), arXiv:2506.23971 (2025).

## Files and build

```text
UMA_server.py       Server and i-PI client
wrappers/           C sources and executables for Gaussian, ORCA, AMBER, CREST
examples/           Inputs for each interface
Makefile            Builds the four C wrappers
requirements.txt    Reference Python dependency versions
LICENSE             MIT license
```

Set up UMA using the [official documentation](https://facebookresearch.github.io/fairchem/) and download its weights. On Linux, compile the wrappers with a C11 compiler and `make`:

```bash
make -B
```

The included binaries target Linux x86-64. Run the server in your UMA environment.

## Server settings

Fill in the model path at the top of `UMA_server.py`:

```python
MODEL_PATH = "/absolute/path/models/uma-s-1p2.pt"  # Required: UMA checkpoint path.
```

The same block contains `PORT`, `DEVICE`, inference switches, batch limits, and Hessian settings. Defaults are port `50423` and device `"cuda"`; use `"cpu"` for CPU inference.

```bash
python UMA_server.py
```

| Option | Effect |
| --- | --- |
| none | Sequential requests; merge, compile, and TF32 off |
| `--batch` | Batch ready energy/gradient requests |
| `--md` | Use `MD_INFERENCE_*`; merge and compile on by default |
| `--batch --md` | Batching with MD settings |
| `--ipi` | Connect to i-PI using MD settings |
| `--verbose` | Print request timings |

TF32 defaults to off in all modes. Enable it manually with `INFERENCE_TF32` or, for MD/i-PI, `MD_INFERENCE_TF32`. Merge and compile improve performance; merging requires fixed composition, charge, and spin, and compilation adds first-run overhead.

**Batching improves GPU utilization and can significantly increase throughput.** Larger batches consume more memory: adjust `RAW_BATCH_MAX_SIZE` (default 32 requests) and `RAW_BATCH_MAX_ATOMS` (2048 atoms) to fit the GPU. CUDA out-of-memory errors trigger batch splitting and retries. Hessian displacement batches use the separate `HESSIAN_BATCH_*` settings.

C wrappers connect to `127.0.0.1:50423`. If you change the server's `PORT`, set the client environment variable `FASTUMA_PORT` to match.

## Gaussian

Set `External` in `examples/gaussian/water.gjf` to the absolute path of `wrappers/gaussian/gaussian_uma`:

```bash
cd examples/gaussian
g16 < water.gjf > water.log
```

The wrapper returns energy, gradients, and finite-difference Hessians. Gaussian dipole and polarizability fields contain zero placeholders.

## ORCA ExtOpt

Set `ProgExt` in `examples/orca/water.inp` to the absolute path of `wrappers/orca/orca_uma`, then run ORCA:

```bash
cd examples/orca
/absolute/path/to/orca water.inp > water.out
```

See the [ORCA ExtOpt manual](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html#external-optimizers).

## AMBER: a fake ORCA written in C

`wrappers/amber/orca` impersonates ORCA for Sander and returns UMA energy and gradients for the QM region.

The [example input](examples/amber/qmmm_nvt.in) uses `ifqnt=1`, `qm_theory='EXTERN'`, and **mechanical embedding** (`qmmm_int=5`): AMBER handles QM–MM electrostatics with topology charges and the classical interactions; UMA evaluates the QM region. Set `qmmask`, `qmcharge`, and `spin` for your system.

Set `use_template=1` in `&orc` and place `orc_job.tpl` in the working directory. Its `! HF STO-3G` line is a placeholder. Sander adds coordinates and the gradient request. See the [AMBER 24 manual, Sections 10.1.6 and 10.2.6.4](https://ambermd.org/doc12/Amber24.pdf).

Put the fake ORCA first in the AMBER environment's `PATH`:

```bash
export PATH="/absolute/path/UMA-CBridge/wrappers/amber:$PATH"
command -v orca
```

The result should point to `wrappers/amber/orca`. Run Sander with your topology and restart:

```bash
cd examples/amber
sander -O -i qmmm_nvt.in -o qmmm_nvt.out \
  -p /absolute/path/system.prmtop -c /absolute/path/equilibrated.rst7 \
  -r qmmm_nvt.rst7 -x qmmm_nvt.nc -inf qmmm_nvt.mdinfo
```

## CREST: a fake ORCA written in C

`wrappers/crest/orca` impersonates ORCA for CREST and supplies UMA energy and gradients for conformational sampling.

The example runs a standard iMTD-GC search on pentane. Edit the two paths in [sampling.toml](examples/crest/sampling.toml): `orca_cmd` points to the C wrapper and `orca_template` to the supplied template. Then run:

```bash
cd examples/crest
crest sampling.toml
```

Conformers are written to `crest_conformers.xyz`. See the [CREST documentation](https://crest-lab.github.io/crest-docs/page/examples/orca.html).

## i-PI

Install i-PI using its [official guide](https://docs.ipi-code.org/getting-started.html). i-PI runs the dynamics; UMA-CBridge supplies energy and forces.

In the i-PI environment, from the project root:

```bash
cd examples/ipi
i-pi methanol.xml
```

In the UMA environment, from the project root:

```bash
python UMA_server.py --ipi \
  --ipi-unix fastuma --ipi-structure examples/ipi/methanol.xyz \
  --ipi-charge 0 --ipi-multiplicity 1
```

The example is 32-bead NVT at 300 K with a 0.25 fs step and PILE-G thermostat. **Batching is enabled in the XML** with `batch_size=32`; run with `--ipi` alone. Adjust the XML batch size to fit GPU memory; `--ipi-max-batch` sets the client limit (default 32).

`--ipi-structure` supplies elements in the same atom order as i-PI. The example uses a UNIX socket named `fastuma`. For TCP, use XML `mode="inet"` and `--ipi-host HOST`; both sides default to port `65535` when the port is omitted. This is separate from the wrapper server's `PORT`. For periodic systems, enable XML PBC and pass `--ipi-pbc`.

## Citation

If UMA-CBridge contributes to your research, please cite this software:

Yuchen Zhou. [UMA-CBridge](https://github.com/RowenaZireael/UMA-CBridge). Citation metadata: [CITATION.cff](CITATION.cff).

Copyright (c) 2026 Yuchen Zhou ([RowenaZireael](https://github.com/RowenaZireael)). [MIT license](LICENSE).

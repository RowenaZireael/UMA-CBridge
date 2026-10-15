# UMA-CBridge

[English](Github/README.md) · 以下路径均相对于 `Github` 项目目录。

UMA-CBridge 为 Meta FAIR Chemistry 的 **UMA 机器学习势**提供 Gaussian、ORCA、AMBER、CREST 和 i-PI 接口，用于几何优化、分子动力学和构象采样。

**C wrapper 与常驻 UMA 服务器显著减少接口调用和通信耗时。** 已有接口如 [ORCA_wrapper_UMA](https://github.com/leforesb/ORCA_wrapper_UMA)、[umas](https://github.com/npe1011/umas) 使用 Python wrapper；本项目用编译后的 C 客户端传输二进制数据，省去每次启动 Python 和导入依赖的开销。作者实测显示，重复调用可节省大量时间。

**AMBER 和 CREST 使用 C 编写的伪 ORCA**：冒充 ORCA 接收调用，以 `.engrad` 文件返回 UMA 能量和梯度。Gaussian 和 ORCA 分别使用 External 和 ExtOpt 接口。`UMA_server.py` 加载一次 UMA，通过 FAIRChem 的 `omol` 任务计算，同时提供 i-PI 力客户端。

UMA：[官方文档](https://facebookresearch.github.io/fairchem/) · [源码](https://github.com/facebookresearch/fairchem) · [模型权重](https://huggingface.co/facebook/UMA)。使用 UMA 请引用 Brandon M. Wood 等人的 [*UMA: A Family of Universal Models for Atoms*](https://arxiv.org/abs/2506.23971)，arXiv:2506.23971（2025）。

## 文件与编译

```text
UMA_server.py       服务器和 i-PI 客户端
wrappers/           Gaussian、ORCA、AMBER、CREST 的 C 源码与可执行文件
examples/           各接口输入示例
Makefile            编译四个 C wrapper
requirements.txt    Python 依赖版本参考
LICENSE             MIT 许可证
```

按 [UMA 官方文档](https://facebookresearch.github.io/fairchem/)部署环境并下载权重。在 Linux 上使用 C11 编译器和 `make` 编译 wrapper：

```bash
make -B
```

附带的可执行文件面向 Linux x86-64。服务器在 UMA 环境中运行。

## 服务器设置

在 `UMA_server.py` 开头填写模型路径：

```python
MODEL_PATH = "/absolute/path/models/uma-s-1p2.pt"  # 必填：UMA 权重文件的绝对路径。
```

同一区域包含端口、设备、推理开关、batch 上限和 Hessian 设置。端口默认 `50423`，设备默认 `"cuda"`；CPU 推理改为 `"cpu"`。

```bash
python UMA_server.py
```

| 参数 | 作用 |
| --- | --- |
| 无参数 | 逐个处理请求；merge、compile、TF32 默认关闭 |
| `--batch` | 批量计算队列中已有的能量／梯度请求 |
| `--md` | 使用 `MD_INFERENCE_*`，默认开启 merge 和 compile |
| `--batch --md` | 批处理配合 MD 设置 |
| `--ipi` | 使用 MD 设置连接 i-PI |
| `--verbose` | 输出请求耗时 |

TF32 在所有模式下默认关闭，通过 `INFERENCE_TF32` 或 MD／i-PI 的 `MD_INFERENCE_TF32` 手动开启。merge 和 compile 用于加速；merge 要求元素组成、电荷和自旋固定，compile 首次运行需要编译时间。

**批量推理能更充分利用显卡，显著提升吞吐量。** 批次越大，显存占用越高。按显存调整 `RAW_BATCH_MAX_SIZE`（默认 32 个请求）和 `RAW_BATCH_MAX_ATOMS`（2048 个原子）；显存不足时，请求批次拆半重试。Hessian 位移结构的批次通过 `HESSIAN_BATCH_*` 单独设置。

C wrapper 默认连接 `127.0.0.1:50423`。修改服务器 `PORT` 时，将客户端环境变量 `FASTUMA_PORT` 设为相同值。

## Gaussian

将 `examples/gaussian/water.gjf` 中的 `External` 改为 `wrappers/gaussian/gaussian_uma` 的绝对路径：

```bash
cd examples/gaussian
g16 < water.gjf > water.log
```

wrapper 返回能量、梯度和有限差分 Hessian。Gaussian 的偶极矩、极化率相关字段以零占位。

## ORCA ExtOpt

将 `examples/orca/water.inp` 中的 `ProgExt` 改为 `wrappers/orca/orca_uma` 的绝对路径，再运行 ORCA：

```bash
cd examples/orca
/absolute/path/to/orca water.inp > water.out
```

参见 [ORCA ExtOpt 手册](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html#external-optimizers)。

## AMBER：C 编写的伪 ORCA

`wrappers/amber/orca` 冒充 ORCA 供 Sander 调用，返回 QM 区域的 UMA 能量和梯度。

[输入示例](Github/examples/amber/qmmm_nvt.in)采用 `ifqnt=1`、`qm_theory='EXTERN'` 和**机械嵌入**（`qmmm_int=5`）：AMBER 用拓扑电荷处理 QM–MM 静电作用和经典相互作用，UMA 计算 QM 区域。按体系设置 `qmmask`、`qmcharge` 和 `spin`。

在 `&orc` 中设置 `use_template=1`，将 `orc_job.tpl` 放入工作目录。其中的 `! HF STO-3G` 是占位内容，Sander 自动添加坐标和梯度请求。参见 [AMBER 24 手册第 10.1.6、10.2.6.4 节](https://ambermd.org/doc12/Amber24.pdf)。

将伪 ORCA 放到 AMBER 运行环境的 `PATH` 最前面：

```bash
export PATH="/absolute/path/UMA-CBridge/wrappers/amber:$PATH"
command -v orca
```

输出应指向 `wrappers/amber/orca`。使用自己的拓扑和重启文件运行：

```bash
cd examples/amber
sander -O -i qmmm_nvt.in -o qmmm_nvt.out \
  -p /absolute/path/system.prmtop -c /absolute/path/equilibrated.rst7 \
  -r qmmm_nvt.rst7 -x qmmm_nvt.nc -inf qmmm_nvt.mdinfo
```

## CREST：C 编写的伪 ORCA

`wrappers/crest/orca` 冒充 ORCA 供 CREST 调用，为构象采样提供 UMA 能量和梯度。

示例对正戊烷执行普通 iMTD-GC 构象搜索。修改 [sampling.toml](Github/examples/crest/sampling.toml) 中的两处路径：`orca_cmd` 指向 C wrapper，`orca_template` 指向附带的模板。然后运行：

```bash
cd examples/crest
crest sampling.toml
```

构象输出到 `crest_conformers.xyz`。参见 [CREST 文档](https://crest-lab.github.io/crest-docs/page/examples/orca.html)。

## i-PI

i-PI 安装参见[官方指南](https://docs.ipi-code.org/getting-started.html)。i-PI 负责动力学，UMA-CBridge 提供能量和力。

在 i-PI 环境中，从项目根目录运行：

```bash
cd examples/ipi
i-pi methanol.xml
```

在 UMA 环境中，从项目根目录运行：

```bash
python UMA_server.py --ipi \
  --ipi-unix fastuma --ipi-structure examples/ipi/methanol.xyz \
  --ipi-charge 0 --ipi-multiplicity 1
```

示例采用 32 珠子、300 K、0.25 fs 步长和 PILE-G 热浴，进行 NVT 模拟。**XML 已通过 `batch_size=32` 开启批处理**，启动时使用 `--ipi` 即可。按显存调整 XML 的批次大小；`--ipi-max-batch` 设置客户端上限，默认 32。

`--ipi-structure` 提供元素信息，原子顺序与 i-PI 保持一致。示例使用名为 `fastuma` 的 UNIX socket。TCP 连接使用 XML 的 `mode="inet"` 和客户端的 `--ipi-host HOST`；省略端口时双方默认 `65535`，与 wrapper 服务器的 `PORT` 分开。周期体系开启 XML 的 PBC，并添加 `--ipi-pbc`。

Copyright (c) 2026 Yuchen Zhou（[RowenaZireael](https://github.com/RowenaZireael)）。[MIT 许可证](Github/LICENSE)。

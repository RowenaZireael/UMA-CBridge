# UMA-CBridge

[English](Github/README.md)

本说明对应同级 `Github` 文件夹中的项目，文中的项目根目录均指 `Github`。

UMA-CBridge 是一套给 **UMA 机器学习原子间势**提供的程序接口，让 Gaussian、ORCA、AMBER、CREST 和 i-PI 能调用 UMA 的能量和梯度，进行几何优化、分子动力学等各程序外部接口支持的计算。UMA 模型由 Meta 的 FAIR Chemistry 团队开发；本项目负责提供连接这些程序的服务器和 wrapper 客户端。

**UMA-CBridge 使用编译后的 C wrapper，显著减少接口调用和通信耗时。** 已有多个开源 UMA 接口采用 Python wrapper，例如 [ORCA_wrapper_UMA](https://github.com/leforesb/ORCA_wrapper_UMA) 和 [umas](https://github.com/npe1011/umas)。本项目为 Gaussian、ORCA、AMBER 和 CREST 提供轻量的 C 可执行程序，通过二进制数值数据与常驻的 UMA 服务器通信，每次调用无需启动 Python 解释器或导入 Python 依赖。作者实测表明，这种方式能节约大量重复接口调用的时间，适合需要频繁计算能量和梯度的几何优化、分子动力学和构象搜索。

服务器只加载一次 UMA 模型，并将它保留在内存中供后续计算使用，通过 FAIRChem 的 `omol` 任务计算分子能量和力。项目支持能量／梯度请求批处理、有限差分 Hessian 分块计算，以及 MoLE 合并和模型编译加速。全部 Python 功能集中在 **`UMA_server.py`**，手动配置位于文件开头；使用 i-PI 时，这个脚本作为力客户端运行。

**提供给 AMBER 和 CREST 的 `orca` 是伪装成 ORCA 的适配程序。** 它们模仿 ORCA 的调用方式和输出，让 AMBER、CREST 把它们当作 ORCA 使用；实际流程是读取结构、请求 UMA 服务器计算，再以 ORCA 兼容的 `.engrad` 文件和输出消息返回 UMA 能量与梯度。这里不执行 ORCA 电子结构计算，也无需安装 ORCA。输入中的方法和基组关键词不改变 UMA 模型。另一个 `wrappers/orca/orca_uma` 客户端则供**真正的 ORCA** 通过 ExtOpt 调用。

UMA 官方资源：[网站与文档](https://fair-chem.github.io/)、[FAIRChem 源码](https://github.com/facebookresearch/fairchem)、[UMA 模型下载](https://huggingface.co/facebook/UMA)。发表使用 UMA 得到的研究结果时，请引用 Brandon M. Wood 等人的 *UMA: A Family of Universal Models for Atoms*，arXiv:2506.23971（2025）：[论文](https://arxiv.org/abs/2506.23971)、[DOI](https://doi.org/10.48550/arXiv.2506.23971)。

## 目录

```text
UMA_server.py                    服务器和 i-PI 客户端，唯一的 Python 脚本
requirements.txt                 Python 依赖版本参考
Makefile                         编译全部四个 C 客户端
wrappers/
  common/fastuma_protocol.h       共用的 socket 协议实现
  gaussian/gaussian_uma.c         Gaussian External 源码
  gaussian/gaussian_uma           Gaussian External 可执行文件
  orca/orca_uma.c                 ORCA ExtOpt 源码
  orca/orca_uma                   ORCA ExtOpt 可执行文件
  amber/orca.c                   AMBER 伪 ORCA 适配器源码
  amber/orca                     AMBER 伪 ORCA 可执行文件
  crest/orca.c                   CREST 伪 ORCA 适配器源码
  crest/orca                     CREST 伪 ORCA 可执行文件
examples/                        各接口的小型输入示例
LICENSE                          本项目的 MIT 许可证
```

## 编译 C wrapper

请按照 [FAIRChem／UMA 官方文档](https://facebookresearch.github.io/fairchem/)部署 UMA 环境，并从 [UMA 官方模型仓库](https://huggingface.co/facebook/UMA)获取权重。`UMA_server.py` 在该环境中运行。本项目提供程序接口，不包含模型权重。

编译四个 C wrapper 需要 Linux、支持 C11 的编译器（GCC 或 Clang）和 `make`。在项目根目录执行：

```bash
make -B
```

附带的可执行文件为 Linux x86-64 构建。`make -B` 会在本机重新编译全部四个 wrapper；`make clean` 删除这些可执行文件。C wrapper 本身不需要 Python 环境。

## 配置和启动

打开 `UMA_server.py` 开头的配置区，直接填写模型路径：

```python
MODEL_PATH = "/absolute/path/models/uma-s-1p2.pt"  # 必填：本机 UMA 模型权重文件（.pt）的绝对路径。
```

`MODEL_PATH` 是唯一必填项，发布脚本中留空，启动前需填写。其后依次为端口 `PORT`、设备 `DEVICE`、普通模式推理设置、MD／i-PI 推理设置、队列批处理上限和四个 Hessian 参数。

```bash
python UMA_server.py
```

启用队列批处理：

```bash
OMP_NUM_THREADS=4 python UMA_server.py --batch
```

**批量推理能更充分地利用显卡，显著提升计算吞吐量，尤其适合单个结构较小的任务。** 开启 `--batch` 后，服务器将多个已经就绪的能量／梯度请求一起计算，让并发客户端任务高效共享同一张显卡。配套 i-PI 示例通过 XML 的 `batch_size` 将 32 个珠子批量计算；Hessian 则单独对位移结构进行批量推理。

批次越大，显存占用也越高，应根据显卡可用显存和每个结构的原子数调整。wrapper 请求由 `RAW_BATCH_MAX_SIZE` 限制每批任务数，`RAW_BATCH_MAX_ATOMS` 控制每批总原子数预算；遇到 CUDA 显存不足时，服务器将该请求批次拆半重试。i-PI 的批次在 XML 中调整，Hessian 的批次则通过脚本开头的 `HESSIAN_BATCH_*` 设置调整。

服务器默认监听 `127.0.0.1:50423`。客户端计算期间保持服务器运行，Ctrl+C 停止服务。

| 启动参数 | 行为 |
| --- | --- |
| 无参数 | 逐个处理请求；TF32、MoLE 合并、编译默认关闭 |
| `--batch` | 将队列中已有的能量/梯度请求一起计算 |
| `--md` | 使用代码开头的 `MD_INFERENCE_*` 配置，适合固定体系 |
| `--batch --md` | 批处理与 MD 配置同时使用 |
| `--ipi` | 使用 MD 配置，连接 i-PI 服务器 |
| `--verbose` | 输出请求或批次的耗时 |

**TF32 是手动开关，目前所有模式下均默认关闭。** `--md` 和 `--ipi` 只是选择下面的 MD 配置，不会自动把 TF32 改为 `True`：

```python
MD_INFERENCE_TF32 = False
MD_INFERENCE_MERGE_MOLE = True
MD_INFERENCE_COMPILE = True
```

要在 `--md`／`--ipi` 下开启 TF32，手动将 `MD_INFERENCE_TF32` 改为 `True`。普通模式保持 TF32 关闭。

MoLE 合并要求元素组成、电荷和自旋保持不变。更换体系时重启服务器；混合体系任务使用未开启合并的配置。首次真实请求会初始化合并权重和编译模型，耗时通常明显更长；批次形状变化也可能触发重新编译。

`PORT` 默认 `50423`；`DEVICE` 默认 `"cuda"`，需要 CPU 推理时改为 `"cpu"`。

四个 C 客户端统一接受 `FASTUMA_HOST`（IPv4 数字地址，默认 `127.0.0.1`）和 `FASTUMA_PORT`（默认 `50423`）。更换本机端口时，修改 `UMA_server.py` 开头的 `PORT`，并将客户端环境变量 `FASTUMA_PORT` 设为相同值。服务器仅绑定回环地址，协议没有认证功能。

## Gaussian

将 Gaussian 的 `External` 命令设为 `wrappers/gaussian/gaussian_uma` 的绝对路径。修改 `examples/gaussian/water.gjf` 中的路径后，用已有的 Gaussian 安装运行：

```bash
cd examples/gaussian
g16 < water.gjf > water.log
```

客户端支持能量、梯度和 Hessian 请求。Hessian 由 UMA 力的中心差分得到；偶极矩、极化率及偶极导数输出为零。该接口提供势能导数，不提供电子性质或红外强度。

## ORCA ExtOpt

修改 `examples/orca/water.inp` 中的 `ProgExt` 路径：

```text
! ExtOpt
%method
  ProgExt "/absolute/path/UMA-CBridge/wrappers/orca/orca_uma"
end
* xyzfile 0 1 water.xyz
```

在示例目录中使用**真正的 ORCA**：

```bash
cd examples/orca
/absolute/path/to/real/orca water.inp > water.out
```

ORCA 生成外部接口文件并调用客户端。客户端读取的是这一接口文件，不是普通 ORCA `.inp`，随后将能量和梯度写入 `<xyz文件基本名>.engrad`。接口说明见 [ORCA ExtOpt 手册](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html#external-optimizers)。

## AMBER 的 ORCA 接口

示例依据 [AMBER 24 手册第 10.1.6、10.2.6.4 节](https://ambermd.org/doc12/Amber24.pdf)的外部 QM 和 ORCA 模板接口编写。使用 `sander` 运行，并将 `orc_job.tpl` 放在计算工作目录中。

`examples/amber/qmmm_nvt.in` 是周期体系 NVT 续跑输入：300 K、Langevin 热浴（`gamma_ln=1.0` ps⁻¹）、0.5 fs 步长，共 100,000 步，即 50 ps。需要准备好拓扑和已平衡的重启文件，后者包含坐标、速度和晶胞（`irest=1, ntx=5`）。示例将第 1 个残基作为中性单重态 QM 区域；使用时按体系调整 `qmmask`、`qmcharge` 和 `spin`。本示例的 QM 区域选取完整分子。MM 中的含氢键使用 SHAKE，QM 键保持柔性（`qmshake=0`）。

接口关键设置为：

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

`qmmm_int=5` 选择机械嵌入：AMBER 用拓扑电荷计算 QM–MM 静电作用并处理经典非键相互作用，UMA 计算 QM 子体系。适配器不计算 MM 点电荷导致的极化及其梯度。QM 静电 Ewald/PME 选项关闭，周期 MM 部分仍由 AMBER 处理。

`orc_job.tpl` 只放方法/基组占位行（`! HF STO-3G`）。适配器忽略该行，实际使用服务器加载的 UMA 权重。**模板中不填写 `EnGrad`、`%pointcharges`、`*xyzfile`、坐标、电荷或多重度。** Sander 自动将计算类型和结构补入 `orc_job.inp`，并写出 `inpfile.xyz`；电荷和多重度来自 `&qmmm`。这两个运行时生成的文件不再作为示例输入分发。

先在项目根目录的独立终端运行 `python UMA_server.py --md`。再在 AMBER 环境中选择适配器并启动模拟：

```bash
export PATH="/absolute/path/UMA-CBridge/wrappers/amber:$PATH"
command -v orca
cd /absolute/path/UMA-CBridge/examples/amber
sander -O -i qmmm_nvt.in -o qmmm_nvt.out \
  -p /absolute/path/system.prmtop -c /absolute/path/equilibrated.rst7 \
  -r qmmm_nvt.rst7 -x qmmm_nvt.nc -inf qmmm_nvt.mdinfo
```

`command -v orca` 应指向 `wrappers/amber/orca`。每个模拟使用独立工作目录，并放入各自的 `orc_job.tpl`。适配器返回 `orc_job.engrad`，ORCA 风格结束标志表示 UMA 计算完成。保持 `dipole=0`，因为此适配器只提供能量和梯度。

## CREST 的 ORCA 接口

修改 `examples/crest/water.toml` 中的两处绝对路径，再运行：

```bash
cd examples/crest
crest water.toml
```

示例使用 `method="orca"`，`orca_cmd` 指向 `wrappers/crest/orca`，`orca_template` 指向最小模板 `orca_template.inp`。适配器接受内嵌 `* xyz` 坐标和 `* xyzfile`，在输入文件旁生成 `.engrad`，通过原子替换更新结果。

示例执行几何优化（`runtype="ancopt"`）。构象搜索可改为 `runtype="imtd-gc"`，再设置适合体系的采样参数。CREST 3.0.2 的 ORCA 接口可能在多线程间复用工作文件路径，因此该版本保持 `threads=1`。多个独立 CREST 作业可使用各自的工作目录，共享 `--batch` 服务器。各版本支持的选项参见 [CREST ORCA 接口文档](https://crest-lab.github.io/crest-docs/page/examples/orca.html)。

可选诊断设置：`CREST_ORCA_LOG=/absolute/path/calls.log` 记录调用；`CREST_ORCA_FSYNC=1` 在替换梯度文件前同步写盘。它们不改变模型配置。

## i-PI

此模式中 **i-PI 是服务器，UMA-CBridge 是力客户端**。积分器、质量、恒温器和环聚合物弹簧由 i-PI 负责；UMA-CBridge 为各个独立结构返回物理势能、力和 virial。

i-PI 的安装请参考 [i-PI 官方指南](https://docs.ipi-code.org/getting-started.html)。

终端一，在已有的 i-PI 环境中，从项目根目录开始：

```bash
cd examples/ipi
i-pi methanol.xml
```

终端二，在已有的 UMA 环境中，从项目根目录开始，先配置好 `MODEL_PATH`：

```bash
OMP_NUM_THREADS=4 python UMA_server.py --ipi \
  --ipi-unix fastuma --ipi-structure examples/ipi/methanol.xyz \
  --ipi-charge 0 --ipi-multiplicity 1
```

每个作业通过 `--ipi-structure` 指定结构，可以直接使用 i-PI 初始化时读取的同一个 XYZ 文件。随附孤立甲醇示例采用 32 个珠子、300 K、0.25 fs 步长，以及 PILE-G 热浴（质心耦合时间 100 fs，`pile_lambda=1.0`）。固定晶胞、不加压强耦合，执行 NVT，共 100,000 步，即 25 ps。建议先将前 5 ps 用于平衡；若观测量仍有趋势，继续延长平衡，并针对目标观测量检查珠子数和采样收敛。性质每 10 步输出，质心轨迹每 100 步输出，重启文件每 1,000 步保存。PILE 参数说明见 [i-PI 热浴设置](https://docs.ipi-code.org/input-tags.html)。首次请求的编译可能需要数分钟，XML 设置了 1800 秒的计算超时。

模板和 i-PI 坐标必须使用完全相同的原子顺序。socket 协议不传递元素、质量、电荷、自旋或 PBC 标记。电荷和多重度由客户端指定，质量在 i-PI 中设置。XML 的 `<parameters>` 保持为空；使用活动子体系时，模板必须与该子体系的原子顺序完全对应。

配套 i-PI 示例默认开启批处理：XML 中已设置 `<batch_size>32</batch_size>`，无需添加 `--batch`。实际批次大小由 i-PI XML 控制；`--ipi-max-batch` 仅设置允许的上限，默认 32。运行中批次大小保持固定。批处理保留默认的消息合并设置，不使用 `matching=lock`。较旧的 i-PI 可去掉 `batch_size`，逐个请求计算。协议说明见 [i-PI socket 文档](https://docs.ipi-code.org/distributed.html)。

UNIX 地址 `fastuma` 对应 `/tmp/ipi_fastuma`，也可直接提供 socket 绝对路径。使用 TCP 时，将 XML 改为 `mode="inet"` 并设置对应地址，客户端传入 `--ipi-host HOST`。XML 不写 `<port>`、命令行不传 `--ipi-port` 时，双方均使用默认端口 `65535`。`--ipi-connect-timeout` 默认为 120 秒；`--ipi-timeout 0` 表示 socket 读写不限时。

孤立分子保持 XML `pbc="False"`，不加 `--ipi-pbc`；virial 根据居中坐标与力计算。三维周期体系添加 `--ipi-pbc` 并相应设置 XML，virial 使用 `-V × stress`。`omol` 的应力由能量导出，未用应力标签拟合，涉及压力的应用需要针对具体体系验证。

## 单位与适用范围

原始协议客户端发送原子序数及 float32 的 Å 坐标；返回 float64 的 Hartree 能量、Hartree/Bohr 梯度，以及按需计算的 Hartree/Bohr² Hessian 下三角。原始协议沿用已有换算常数；i-PI 使用 float64 的 Bohr/Hartree 数据并通过 ASE 换算。梯度与力互为相反数。

原始接口计算孤立结构。外部电场、点电荷、电子密度及任意 ORCA/Gaussian 方法关键词不会作为 UMA 输入。请选用适合研究体系的模型权重和任务。

Copyright (c) 2026 Yuchen Zhou（[RowenaZireael](https://github.com/RowenaZireael)）。本项目代码使用 [MIT 许可证](Github/LICENSE)。UMA 权重及第三方程序分别遵循各自的许可证。

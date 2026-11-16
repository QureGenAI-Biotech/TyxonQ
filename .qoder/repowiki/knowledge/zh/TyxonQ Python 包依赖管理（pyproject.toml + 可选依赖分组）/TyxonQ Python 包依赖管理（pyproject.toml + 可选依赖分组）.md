---
kind: dependency_management
name: TyxonQ Python 包依赖管理（pyproject.toml + 可选依赖分组）
category: dependency_management
scope:
    - '**'
source_files:
    - pyproject.toml
    - docs/requirements-doc.txt
    - examples/SMILES-TyxonQ/requirements.txt
    - README.md
    - QCOS_CHANGES.md
---

## 1. 使用的系统/方法

- **打包与发布**：基于 `setuptools.build_meta`（`[build-system]`），通过 `pyproject.toml` 声明元数据、依赖与构建配置，无独立的 `setup.py`。
- **依赖声明**：全部集中在根目录的 `pyproject.toml`，使用 PEP 621 `[project.dependencies]` 与 `[project.optional-dependencies]` 两个字段。
- **Python 版本约束**：`requires-python = ">=3.10,<3.13"`，限定 3.10–3.12。
- **可选功能域（extras）**：按能力域拆分为 `sbd`、`geomopt`、`spin`、`lqcloud`、`guodun`、`md`、`dev` 等 extras，用户通过 `pip install tyxonq[extra]` 按需安装。
- **文档构建依赖**：单独维护 `docs/requirements-doc.txt`，由 Sphinx 文档站点使用。
- **示例子项目**：`examples/SMILES-TyxonQ/requirements.txt` 为独立示例脚本的轻量依赖清单；其余 examples 不附带 requirements 文件，依赖核心包。
- **无锁定文件**：仓库未提交 `poetry.lock`、`uv.lock`、`Pipfile.lock` 或 `requirements.txt`（除示例外），也未见 `vendor/` 目录或私有 PyPI 源配置。
- **CI/自动化更新**：未发现 GitHub Actions / CI 中自动升级依赖的配置，也没有 `dependabot.yml`、`renovate.json` 等自动化更新工具。

## 2. 关键文件

- `pyproject.toml` — 包名、版本、运行时依赖、可选依赖、构建后端、pytest 配置。
- `docs/requirements-doc.txt` — 文档站点的额外依赖。
- `examples/SMILES-TyxonQ/requirements.txt` — 该示例子项目的依赖清单。
- `src/tyxonq.egg-info/requires.txt`、`dependency_links.txt`、`SOURCES.txt` — setuptools 生成的 egg-info 产物，非源码控制。

## 3. 架构与约定

- **核心运行时依赖**（`dependencies`）：包含量子化学与数值栈（`openfermion>=1.7.1`、`openfermionpyscf>=0.5`、`pyscf>=2.10`、`renormalizer>=0.0.11`）、线性代数（`numpy>=2.0.2`、`scipy>=1.13.1`、`opt-einsum>=3.4.0`）、符号计算（`symengine>=0.14.1`、`sympy>=1.14.0`）、深度学习（`torch>=2.7.1`、`torchvision>=0.22`）、量子生态（`qiskit>=2.1.0`）以及通用工具（`requests>=2.32.4`、`python-dotenv>=1.1.1`、`ruff>=0.12.2`、`pytest>=8.4`）。这些均为“松下限”（仅 `>=`），没有上限锁。
- **可选依赖分组**：
  - `sbd`：`pybind11>=2.12,<4`、`mpi4py>=4.0,<5`（GPU SBD 扩展编译）。
  - `geomopt`：`ase>=3.23`、`sella>=2.3,<3`（几何优化）。
  - `spin`：`quimb>=1.11,<2`、`cotengrust>=0.2,<0.3`、`scipy>=1.14`、`matplotlib>=3.8`。
  - `lqcloud`：`lqcloud==0.4.2`（精确 pin）。
  - `guodun`：`cqlib==1.3.11`（精确 pin）。
  - `md`：`ase>=3.23`、`openmm>=8.1`、`openmmml>=1.7`（分子动力学接口）。
  - `dev`：`jupyter`、`pytest-lazy-fixtures`、`build`、`twine` 等开发工具。
- **构建产物打包**：`[tool.setuptools.package-data]` 将 `*.txt`、`*.md`、`*.rst` 以及 C++/CUDA 源码（`*.cpp`、`*.cuh`、`*.patch`、`SOURCES.json`）打入 wheel，说明 SBD 扩展在打包时随源码分发而非预编译二进制。
- **第三方闭源依赖处理**：`README.md` 与 `QCOS_CHANGES.md` 中出现 `pip install wuyue-1.0-py3-none-any.whl` 的手动安装说明，表明存在不在 PyPI 上的本地 wheel 依赖，需离线安装。
- **版本策略**：核心依赖普遍采用 `>=` 下限，少数 vendor SDK（`lqcloud==0.4.2`、`cqlib==1.3.11`）使用精确 pin，以规避上游 ABI/API 不兼容。

## 4. 约定与约束

- **单一依赖入口**：所有包级依赖统一声明于 `pyproject.toml` 的 `[project.dependencies]` 与 `[project.optional-dependencies]`，未在代码中动态拼接 pip 命令来拉取依赖。
- **可选功能通过 extras 暴露**：新增外部库应添加到对应的 `[project.optional-dependencies]` 分组，并通过 `pip install tyxonq[extra]` 安装，而不是加入核心 `dependencies`。
- **Vendor SDK 精确 pin**：`lqcloud==0.4.2`、`cqlib==1.3.11` 使用 `==` 锁定，避免上游变更破坏硬件驱动兼容性。
- **无锁定文件**：仓库未提交任何 lockfile（如 `poetry.lock`、`uv.lock`、`requirements.txt` 主清单），因此依赖版本仅在 `pyproject.toml` 中声明，实际解析由 pip/setuptools 在每次安装时进行。
- **无私有注册表/GOPRIVATE**：未发现 `.pypirc`、`pip.conf`、`poetry.toml` 中的 `repositories` 或 `source` 配置，也未见 GOPRIVATE 等 Go 私有源设置（本项目为纯 Python）。
- **文档与示例分离**：文档依赖走 `docs/requirements-doc.txt`，示例子项目 `examples/SMILES-TyxonQ` 有独立 `requirements.txt`，但其他 examples 复用核心包依赖，无各自清单。
- **手动 wheel 安装流程**：对不在 PyPI 的依赖（如 `wuyue`），README 明确给出 `pip install wuyue-1.0-py3-none-any.whl` 的步骤，作为官方支持的补充安装方式。
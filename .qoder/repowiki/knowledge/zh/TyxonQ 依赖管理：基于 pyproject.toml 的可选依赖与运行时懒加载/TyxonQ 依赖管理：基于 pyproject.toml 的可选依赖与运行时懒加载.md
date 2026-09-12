---
kind: dependency_management
name: TyxonQ 依赖管理：基于 pyproject.toml 的可选依赖与运行时懒加载
category: dependency_management
scope:
    - '**'
source_files:
    - pyproject.toml
    - docs/requirements-doc.txt
    - examples/SMILES-TyxonQ/requirements.txt
    - examples/gqe_drug_design/requirements.txt
    - src/tyxonq/__init__.py
    - src/tyxonq/about.py
    - .pypirc
---

## 1. 使用的系统与方法

本仓库采用 **Python 原生包管理**，以 `pyproject.toml`（PEP 621）作为唯一声明式依赖入口，构建后端使用 `setuptools.build_meta`。没有发现 `requirements.txt` 作为安装清单（仅示例脚本附带独立的 `requirements.txt` 用于演示环境），也没有 `poetry.lock`、`pipenv`、`conda` 或 `go.mod`/`package.json` 等其它语言依赖文件。

## 2. 关键文件

- `pyproject.toml`：核心依赖、可选依赖分组、构建系统与 pytest 配置。
- `docs/requirements-doc.txt`：文档站点（Sphinx + i18n）的独立依赖清单。
- `examples/SMILES-TyxonQ/requirements.txt`、`examples/gqe_drug_design/requirements.txt`：示例脚本各自声明的轻量依赖。
- `src/tyxonq/__init__.py`、`src/tyxonq/about.py`：通过 `try/except ImportError` 实现可选依赖的运行时懒加载。
- `.pypirc`：PyPI 发布凭据（表明包通过标准 PyPI 渠道分发）。

## 3. 架构与约定

### 3.1 依赖分层与可选特性

`pyproject.toml` 将依赖划分为两层：

- **核心依赖**（`dependencies` 列表）：所有用户安装 `tyxonq` 都会获得，包括 `numpy>=2.0.2`、`scipy>=1.13.1`、`torch>=2.7.1`、`qiskit>=2.1.0`、`pyscf>=2.10`、`openfermion>=1.7.1`、`requests>=2.32.4`、`symengine`、`sympy`、`opt-einsum`、`python-dotenv`、`ruff`、`pytest` 等。
- **可选依赖**（`project.optional-dependencies`）：按功能域拆分，避免默认安装体积膨胀：
  - `lqcloud`：`lqcloud==0.4.2`（锁定到精确版本）。
  - `guodun`：`cqlib==1.3.11`（锁定到精确版本）。
  - `md`：`ase>=3.23`、`openmm>=8.1`、`openmmml>=1.7`（分子动力学接口）。
  - `dev`：`jupyter`、`pytest-lazy-fixtures`、`build`、`twine` 等开发工具。

注释中明确说明 i-PI 与 MDI 为“可选专线”，需单独 `pip install ipi` / `pip install 'pymdi>=1.4'`，不在包内声明。

### 3.2 运行时懒加载（Optional Import）

核心包在导入时不会强制加载重型可选依赖。`src/tyxonq/__init__.py` 和 `src/tyxonq/about.py` 大量使用 `try: import ... except ImportError:` 模式，仅在真正调用相关功能时才触发导入。这使得用户可以只安装核心依赖而不安装 `torch`、`qiskit`、`pyscf` 等可选组件。

### 3.3 构建与发布

- 构建系统：`[build-system] requires = ["setuptools>=65", "wheel"]`，`build-backend = "setuptools.build_meta"`。
- 包发现：`[tool.setuptools.packages.find] where = ["src"]`，源码位于 `src/tyxonq/`。
- 包数据：包含 `*.txt`、`*.md`、`*.rst` 资源文件。
- 发布凭据：`.pypirc` 存在，配合 `twine`（在 `dev` 可选依赖中声明）进行上传。

### 3.4 文档依赖隔离

`docs/requirements-doc.txt` 完全独立于主包依赖，专门用于 Sphinx 文档构建，包含 `sphinx>=7.0.0`、`pydata-sphinx-theme`、`sphinx-intl` 等，并通过 `tyxonq >=0.9` 引用已发布的包而非本地源码。

### 3.5 示例依赖

每个示例目录可自带 `requirements.txt`（如 `examples/SMILES-TyxonQ/requirements.txt` 列出 `torch`、`pyscf`、`wandb`；`examples/gqe_drug_design/requirements.txt` 列出 `pennylane`），用于该示例的独立运行环境，不污染主包依赖。

## 4. 约定与约束

- **Python 版本约束**：`requires-python = ">=3.10,<3.13"`，限定 3.10–3.12。
- **可选依赖必须通过 `pip install tyxonq[feature]` 显式启用**：如 `pip install tyxonq[lqcloud]`、`pip install tyxonq[guodun]`、`pip install tyxonq[md]`、`pip install tyxonq[dev]`。
- **硬件厂商 SDK 使用精确版本锁定**：`lqcloud==0.4.2`、`cqlib==1.3.11`，避免上游变更破坏兼容性。
- **核心数值库使用宽松下限**：`numpy>=2.0.2`、`scipy>=1.13.1`、`torch>=2.7.1`，允许向上兼容更新。
- **无 vendoring**：未发现 `vendor/` 目录或第三方源码嵌入，所有依赖通过 pip 解析。
- **无私有注册表配置**：未检测到 `setup.cfg`、`pip.conf` 或 `pyproject.toml` 中的 `[index-url]` / `[[tool.poetry.source]]` 等私有源配置。
- **测试依赖与运行依赖混用**：`pytest>=8.4` 同时出现在核心依赖与 `dev` 可选依赖中，测试代码直接依赖 pytest。
- **CI/锁文件缺失**：仓库中未见 `requirements.lock`、`poetry.lock`、`Pipfile.lock` 或 GitHub Actions 中的依赖锁定步骤，依赖版本由 `pyproject.toml` 中的 `>=` 约束控制，非冻结版本。

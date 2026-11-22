---
kind: build_system
name: TyxonQ 构建与打包体系（pyproject + setuptools + Sphinx + pytest）
category: build_system
scope:
    - '**'
source_files:
    - pyproject.toml
    - MANIFEST.in
    - docs/Makefile
    - docs/build_docs.sh
    - .readthedocs.yaml
    - docs/requirements-doc.txt
    - .pypirc
---

## 1. 构建系统概览

TyxonQ 采用纯 Python 包形态，以 `pyproject.toml` 为唯一入口，使用 `setuptools.build_meta` 作为后端，通过 `src/` layout 分发 `tyxonq` 包。测试由 `pytest` 驱动，文档由 Sphinx 构建并通过 Read the Docs 托管。仓库根没有 Makefile、Dockerfile 或 CI 流水线文件（`.github/workflows/` 不存在），因此本地构建是主要交付方式。

## 2. 关键文件

- `pyproject.toml`：项目元数据、依赖、可选 extras、构建后端、pytest 配置。
- `MANIFEST.in`：控制 sdist 中额外包含的文件（LICENSE、README、examples、tests 等）。
- `docs/Makefile`：Sphinx 多语言文档构建入口（html / html-zh / html-ja / latexpdf / epub / gettext / update-po）。
- `docs/build_docs.sh`：封装多语言文档工作流的 Bash 脚本（en|zh|ja|all|workflow|clean 等子命令）。
- `.readthedocs.yaml`：Read the Docs 平台配置（Ubuntu 22.04 + Python 3.11，输出 PDF/EPUB）。
- `docs/requirements-doc.txt`：文档站点专用依赖清单。
- `.pypirc`：PyPI/TestPyPI 发布凭据（含 cqlib、lqcloud 私有索引）。

## 3. 架构与约定

### 3.1 包结构与版本
- 源码位于 `src/tyxonq/`，`[tool.setuptools.packages.find] where = ["src"]` 自动发现。
- 版本号在 `pyproject.toml` 的 `[project] version = "1.4.0"` 处声明，无独立 `__version__.py`；`src/tyxonq/about.py` 可能用于运行时读取。
- Python 版本约束 `requires-python = ">=3.10,<3.13"`，分类器覆盖 3.10/3.11/3.12。

### 3.2 依赖管理
- 核心依赖集中在 `dependencies` 列表（openfermion、pyscf、qiskit、torch、scipy、symengine 等）。
- 可选功能通过 `[project.optional-dependencies]` 拆分为 `sbd`、`geomopt`、`spin`、`lqcloud`、`guodun`、`md`、`dev` 等 extras，用户按需安装。
- `sbd` 额外依赖 pybind11/mpi4py，表明存在 C++ 扩展（`tyxonq.applications.chem.algorithms.sqd.gpu.sbd_ext` 下的 *.cpp/*.cuh 由 package-data 显式纳入）。

### 3.3 打包产物
- 构建后端：`setuptools.build_meta`，要求 `setuptools>=65, wheel`。
- `MANIFEST.in` 将 LICENSE、README、CONTRIBUTING、examples（*.py/*.ipynb/*.md/*.json）、tests（recursive include）纳入 sdist，同时排除 .git*、build、dist、*.egg-info、*.so、*.pyc 等。
- `package-data` 额外把 `*.txt`/`*.md`/`*.rst` 以及 SQD GPU 扩展源文件打入 wheel。

### 3.4 测试体系
- 测试路径由 `[tool.pytest.ini_options] testpaths` 指定五个目录：`tests_core_module`、`tests_examples`、`tests_applications_chem`、`tests_mol_valid`、`tests_applications_spin`。
- 默认运行选项 `-v -ra`，并定义 `live` marker 标记需要访问外部服务的测试（默认跳过）。
- `dev` extra 包含 `pytest-lazy-fixtures`、`build`、`twine` 等开发工具。

### 3.5 文档构建
- `docs/Makefile` 提供 `html`、`html-zh`、`html-ja`、`html-all`、`latexpdf`、`epub`、`gettext`、`update-po`、`linkcheck`、`doctest`、`clean-all` 等目标。
- `docs/build_docs.sh` 封装完整 i18n 工作流：`gettext` → `update-po` → `build_all`，支持 `en|zh|ja|all|workflow|clean|help` 参数。
- Read the Docs 使用 `.readthedocs.yaml`，Python 3.11，Sphinx 配置指向 `doc-tyxonq/source/conf.py`（注意该路径与当前仓库 docs/source/conf.py 不一致，可能是历史遗留或子模块差异），fail_on_warning=false，输出 PDF+EPUB。
- 文档依赖单独维护于 `docs/requirements-doc.txt`（sphinx-autodoc-typehints、pydata-sphinx-theme、sphinx-intl 等）。

### 3.6 发布流程
- 通过 `twine`（`dev` extra）上传到 PyPI/TestPyPI，凭据存放在 `.pypirc` 的 `[pypi]` 和 `[testpypi]` 段。
- 还定义了 `[cqlib]`、`[lqcloud]` 两个私有索引条目，对应框架内硬件提供商的私有包源。
- 未发现 GitHub Actions/GitLab CI 等自动化流水线文件；文档 README 提到 `.github/workflows/docs.yml` 待创建。

## 4. 约定与约束

- **包布局**：源码必须放在 `src/` 下，由 setuptools 自动发现（`where = ["src"]`）。
- **Python 版本**：仅支持 3.10–3.12（`requires-python = ">=3.10,<3.13"`）。
- **可选功能**：新增可选能力应通过 `[project.optional-dependencies]` 新增一个 extra key，并在代码中以条件导入方式加载。
- **C/C++ 扩展**：SQD GPU 扩展的源码（*.cpp/*.cuh/*.patch/SOURCES.json）需通过 `[tool.setuptools.package-data]` 显式声明才能进入 wheel。
- **测试组织**：测试按模块分目录存放（core/examples/applications_chem/mol_valid/applications_spin），新模块应遵循此命名约定。
- **文档多语言**：新增语言需在 `docs/Makefile` 增加 `html-xx` 目标，并在 `docs/build_docs.sh` 中添加对应分支；翻译模板经 `make gettext` 生成后由 `sphinx-intl update` 同步到 `locale/pot/`。
- **发布凭据**：PyPI/TestPyPI 用户名密码硬编码在 `.pypirc`，未使用环境变量或密钥管理服务（安全风险）。
- **CI 缺失**：仓库根无 `.github/` 目录，无自动化构建/测试/发布流水线；`.readthedocs.yaml` 中的 sphinx configuration 路径 (`doc-tyxonq/source/conf.py`) 与当前仓库结构不一致，可能无法直接触发 RTD 构建。

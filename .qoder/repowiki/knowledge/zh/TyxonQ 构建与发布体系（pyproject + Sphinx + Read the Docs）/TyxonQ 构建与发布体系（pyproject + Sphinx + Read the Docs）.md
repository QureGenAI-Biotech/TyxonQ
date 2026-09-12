---
kind: build_system
name: TyxonQ 构建与发布体系（pyproject + Sphinx + Read the Docs）
category: build_system
scope:
    - '**'
source_files:
    - pyproject.toml
    - MANIFEST.in
    - docs/Makefile
    - docs/build_docs.sh
    - .readthedocs.yaml
    - .pypirc
---

## 1. 构建系统概览

TyxonQ 采用现代 Python 打包方案，以仓库根目录的 `pyproject.toml` 为唯一入口，使用 `setuptools.build_meta` 作为后端，通过 `src/` 布局将源码隔离在 `src/tyxonq/` 下。依赖声明集中在 `pyproject.toml` 的 `[project]` 与 `[project.optional-dependencies]` 中，按功能域拆分为可选依赖：`lqcloud`、`guodun`、`md`（分子动力学接口）、`dev`（开发工具链）。Python 版本约束为 `>=3.10,<3.13`，并显式声明支持 3.10/3.11/3.12。

## 2. 核心文件与职责

- `pyproject.toml`：包名、版本号（当前 `1.4.0`）、元数据、全部运行时依赖、可选依赖、pytest 配置（测试路径覆盖 `tests_core_module`、`tests_examples`、`tests_applications_chem`、`tests_mol_valid`）、构建后端与包发现规则（`where = ["src"]`）。
- `MANIFEST.in`：控制 sdist 内容，包含 LICENSE、README.md、CONTRIBUTING.md、`examples/*.py *.ipynb *.md`、`tests/*.py`，同时全局排除 `.pyc`、`.so`、`__pycache__`、`.git*`、`build/`、`dist/`、`*.egg-info`。
- `docs/Makefile`：Sphinx 文档构建入口，提供 `html`、`html-zh`、`html-ja`、`html-all`、`gettext`、`update-po`、`latexpdf`、`epub`、`linkcheck`、`doctest`、`clean-all` 等目标，输出到 `docs/build/{html,html-zh,html-ja}`。
- `docs/build_docs.sh`：多语言文档构建脚本，封装 `make` 调用并提供 `en|zh|ja|all|gettext|update-po|workflow|clean|help` 子命令，默认构建英文文档。
- `.readthedocs.yaml`：Read the Docs 云端文档构建配置，指定 Ubuntu 22.04 + Python 3.11 环境，Sphinx 配置文件位于 `doc-tyxonq/source/conf.py`（注意与本地 `docs/source/conf.py` 分离），生成 PDF 与 EPUB 格式。
- `.pypirc`：PyPI / TestPyPI / cqlib / lqcloud 上传凭据（含用户名与 token），用于 `twine upload` 发布。

## 3. 架构与约定

### 3.1 包构建与分发
- 构建后端：`setuptools.build_meta`，要求 `setuptools>=65` 与 `wheel`。
- 包发现：`tool.setuptools.packages.find` 扫描 `src/` 下的 Python 包。
- 包数据：`tool.setuptools.package-data` 将 `src/tyxonq/*.txt|*.md|*.rst` 随包分发。
- 源分发清单：由 `MANIFEST.in` 精确控制，明确包含示例与测试以便下游用户直接运行。
- 发布流程：通过 `dev` 可选依赖引入 `build` 与 `twine`，配合 `.pypirc` 中的 PyPI/TestPyPI/cqlib/lqcloud 凭据进行上传。

### 3.2 测试执行
- 测试框架：pytest（`pytest>=8.4` 为运行时依赖，`dev` 中额外包含 `pytest-lazy-fixtures`）。
- 测试组织：四个独立目录分别对应核心模块、示例、化学应用、分子验证；通过 `pyproject.toml` 的 `[tool.pytest.ini_options]` 统一注册。
- 标记策略：定义 `live` marker 标记需要访问外部服务的测试，默认跳过，避免 CI 误触发真实硬件/云平台调用。

### 3.3 文档构建
- 本地构建：`docs/Makefile` 提供多语言目标，`docs/build_docs.sh` 封装完整工作流（提取 gettext → 更新 PO → 构建所有语言）。
- 多语言：基于 Sphinx i18n，支持英文、中文（`zh_CN`）、日文（`ja_JP`），翻译模板存放于 `docs/locale/`。
- 云端构建：`.readthedocs.yaml` 指定独立的 `doc-tyxonq/` 子项目路径，与本地 `docs/` 并行存在，可能用于不同分支或镜像站点。

### 3.4 版本管理
- 单一来源：版本号硬编码在 `pyproject.toml` 的 `[project] version = "1.4.0"`，未检测到自动化版本注入脚本。
- 变更日志：`CHANGELOG.md` 记录版本历史，但未见与构建脚本联动的自动 bump 机制。

## 4. 约定与约束

- **源码隔离**：所有 Python 代码必须放在 `src/tyxonq/` 下，通过 `setuptools.packages.find(where=["src"])` 自动发现，禁止在仓库根目录放置顶层 Python 包。
- **可选依赖拆分**：平台相关依赖（`lqcloud`、`guodun`）与领域扩展（`md`）均通过 optional-dependencies 暴露，核心包不强制安装这些重型依赖。
- **测试默认跳过 live 用例**：通过 pytest marker 机制，任何访问外部服务（云平台、硬件设备）的测试必须打上 `@pytest.mark.live`，否则 CI 默认跳过。
- **文档多语言同步**：新增可翻译字符串需先经 `make gettext` 提取 POT，再通过 `make update-po` 更新 zh_CN/ja_JP 的 PO 文件，最后用 `make html-all` 构建。
- **sdist 内容受 MANIFEST.in 约束**：构建产物会包含示例与测试，但严格排除 `.pyc`、`.so`、`.git*`、`build/`、`dist/`、`*.egg-info` 等临时文件。
- **CI/CD 缺失**：仓库未发现 `.github/workflows/` 或其他 CI 配置文件，发布与测试流程目前依赖本地脚本与 `.pypirc` 手动执行。
- **Read the Docs 路径差异**：云端构建指向 `doc-tyxonq/source/conf.py`，与本地 `docs/source/conf.py` 分离，需注意两者配置一致性。
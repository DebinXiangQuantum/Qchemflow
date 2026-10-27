# Qchemflow

Qchemflow contains a Python-embedded quantum chemistry language, QCL, and the
ChemFlow runtime and Agent integration it uses. The repository name is
**Qchemflow**; the Python import names are `qcl` and `chemflow`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `qcl/` | Python frontend, Rust compiler/runtime, device adapters, examples and tests |
| `src/chemflow/` | Chemistry implementations, execution support, CLI, Agent and MCP server |
| `plugin/` | Source templates for Agent plugins, roles, skills and integrity hooks |
| `examples/` | Small ChemFlow runtime examples used by the smoke checks |
| `docs/spec/qcl-design.md` | QCL design specification and acceptance criteria |

QCL reuses ChemFlow's chemistry implementations and artifact handling. The
current implementation has working CPU reference workflows and explicit GPU
and QPU placement paths. Primitive and application coverage is still being
expanded against the design specification; an example in `qcl/apps/` is not a
claim that every device supports that operation. Unsupported placements fail
with a diagnostic instead of silently moving to another device.

## Install and run

Python 3.11 and a Rust toolchain are required. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e . pytest ruff maturin
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --manifest-path qcl/Cargo.toml
OMP_NUM_THREADS=1 .venv/bin/python qcl/apps/h2_ground.py
```

`qcl/apps/h2_ground.py` compiles and runs an H2 full-CI energy workflow on the
local CPU. Other examples include `h2_mp2.py`, `h2_gradient.py`,
`h2_pauli_estimate.py` and `agent_h2.py`. Each workflow places compute operations
explicitly with `.to(device)`.

For a CUDA GPU host, install the optional GPU dependencies with
`.venv/bin/python -m pip install -e '.[gpu]'`. GPU execution requires a compatible
NVIDIA driver and supported backend for the selected primitive. For QPU
simulation, install `.[quantum]`; the local cluster provides a simulator QPU.
For the lqcloud MQ02 service, install `.[qpu]` and set `LQCLOUD_API_KEY` in the
execution environment. Cloud QPU submission is controlled by a QCL policy
file. Monetary budgets and approval thresholds apply to metered cloud QPUs,
not local simulators or other non-billable devices. Never commit credentials or
generated run records.

## ChemFlow Agent and plugins

The Agent accepts a signed ChemFlow task and a trusted public key. Select the
QCL engine explicitly when running a reviewed QCL program:

```bash
.venv/bin/chemflow agent run \
  --task /path/to/signed-task.json \
  --public-key /path/to/public.pem \
  --engine qcl \
  --program qcl/apps/agent_h2.py \
  --out /path/to/run-output
```

The `plugin/` directory is the single source for Agent roles, skills and
hooks. Build an adapter for `codex`, `agent-plugins`, `claude-code`, `goose`,
`hermes` or `pi` with:

```bash
.venv/bin/chemflow plugin build \
  --target codex \
  --out /path/to/plugin-output \
  --public-key /path/to/public.pem \
  --task /path/to/signed-task.json
```

The plugin builder records absolute paths to the installed `chemflow-mcp` and
`chemflow-hook` commands. Build the adapter in the environment where it will
run. Use `chemflow --help` and `chemflow agent --help` for the remaining CLI
options.

## Checks

Install the `quantum`, `reactivity` and `paths` extras before running the full
CPU test suite; they provide the QPU simulator, RDKit, tblite, Sella and related
test backends.

```bash
.venv/bin/python -m pip install -e '.[quantum,reactivity,paths]'
cd qcl && cargo fmt --all --check && PYO3_PYTHON="$PWD/../.venv/bin/python" cargo test --workspace --locked
cd ..
.venv/bin/python -m pytest qcl/tests tests/test_qcl_agent.py tests/test_kernel.py tests/test_physics.py tests/test_p2_build.py -q
```

The included GitHub Actions workflow runs the CPU and Agent checks. GPU and
cloud QPU checks require the corresponding hardware or service credentials.

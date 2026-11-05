# Stubs must be installed non-editable so glmm-stubs/ lands in site-packages
export UV_NO_EDITABLE := "1"

# Compare stubs against the installed glmm runtime
stubtest *args:
    uv run python -m mypy.stubtest glmm {{args}}

# Type-check upstream glmm tests against the stubs
check-upstream:
    uv run pyrefly check --config pyrefly.toml ../glmm/python/tests

# Record ALL current upstream errors in upstream-baseline.json (review `just check-upstream` first!)
update-baseline:
    -uv run pyrefly check --config pyrefly.toml ../glmm/python/tests --update-baseline
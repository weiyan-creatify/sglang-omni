# Troubleshooting

## Regional PyPI mirrors

**Symptom.** A configured PyPI mirror may be missing a required package version or lag
behind PyPI. For example, if the mirror does not provide
`mooncake-transfer-engine-cuda13>=0.3.10`, installation fails to resolve the
requirement in [`pyproject.toml`](https://github.com/sgl-project/sglang-omni/blob/main/pyproject.toml). This is an index
availability issue; the mirror does not have the required version.

**Check.** If uv reports that the requirement was not found in the package
registry, check the configured mirror for the required wheel. To see whether
PyPI has a wheel for your environment, use the download command below.

**Fix.** Download that one wheel from PyPI, then use `--find-links` while keeping your
existing uv mirror configuration for the other packages. With the Python 3.12
virtual environment from the {ref}`installation guide <install-from-source>`
active, install pip into it so the downloaded wheel matches that environment:

```bash
uv pip install pip
python -m pip download --no-deps --index-url https://pypi.org/simple \
    "mooncake-transfer-engine-cuda13>=0.3.10" -d ./wheels
uv pip install --prerelease=allow --find-links ./wheels -e .   # or "sglang-omni==<version>" in place of -e .
```

## Expected output of uv pip check

**Symptom.** A CUDA installation finishes, but the dependency check reports
version or platform conflicts.

**Check.** With the SGLang-Omni environment active, run:

```bash
uv pip check
```

**Interpret the output.** `uv pip check` and `python -m pip check` compare the requirements every installed distribution declares with the versions installed next to it. The CUDA install paths leave a few declared ranges unmet on purpose, so the lines in this table are expected and the code they concern runs. Investigate any other line before serving a model.

| Install path | Reported line | Why the runtime still works |
| --- | --- | --- |
| Every CUDA path | `grpcio-health-checking` and `grpcio-reflection` require `protobuf>=7.35.1,<8.0.0`, protobuf 6.x is installed | SGLang pulls both through `smg-grpc-servicer`, and `tool.uv.override-dependencies` in `pyproject.toml` keeps protobuf in the 6.x series, so `import grpc_health.v1.health_pb2` fails with a protobuf `VersionError` (gencode 7.35.1, runtime 6.33.6). Neither SGLang-Omni nor any module of the installed `sglang` package imports `grpc_health` or `grpc_reflection`. |
| After `uv pip install --no-deps qwen-tts==0.1.1` | `qwen-tts` requires `transformers==4.57.3` | The project runs Transformers 5.19.0, and `sglang_omni/models/qwen3_tts/compat.py` shims the API differences, see the [Qwen3-TTS cookbook](../cookbook/qwen3_tts.md#prerequisites). |
| Same | `qwen-tts` requires `accelerate==1.12.0` | `qwen-tts` 0.1.1 never imports `accelerate`. |
| Same | `qwen-tts` requires `onnxruntime`, which is not installed | The CUDA path installs `onnxruntime-gpu`, which provides the `onnxruntime` module the speech tokenizer imports. |
| `[fishaudio]` extra and the CI image | `descript-audiotools` requires `protobuf<3.20,>=3.9.2` | The override keeps protobuf 6.x. `audiotools` reaches protobuf only through `torch.utils.tensorboard`, which the installed TensorBoard supports, and the CI import probe imports `dac` on every run. |
| Every CUDA path | `decord 0.6.0 is not supported on this platform` (pip) or `The package decord was built for a different platform` (uv) | The wheel is published as `py3-none-manylinux2010_x86_64`, but its WHEEL file records the tag `cp36-cp36m`. decord loads `libdecord.so` through ctypes, and the MiniCPM-o video frontend decodes with it on Python 3.12. |

The two tools word the lines differently, and `uv pip check` exits with status 1 whenever it prints one, so its exit status is not a gate. A source checkout has a script that runs the same comparison with the interpreter it is started with and fails only on a line outside this table. The PyPI package does not include it, so run it from the root of the checkout, for example after {ref}`installing from source <install-from-source>`, with the environment's Python:

```bash
cd sglang-omni   # the root of the source checkout
python .github/scripts/check_omni_installed_requirements.py
```

The install commands keep `--prerelease=allow` from the releases that pinned `cuda-tile==1.6.0rc5`; SGLang 0.5.23 pins no pre-release package and the CUDA path resolves without the flag. With the flag, uv may take a pre-release of any package when one is newer than the latest release on PyPI, which is why `uv pip freeze` can list development or release-candidate versions of packages such as `einops`, `safetensors` or `grpcio`.

SGLang 0.5.23 ships `manylinux_2_34` wheels, so the CUDA path needs glibc 2.34 or newer, as on Ubuntu 22.04. On an older system uv stops with `sglang==0.5.23 has no wheels with a matching platform tag`.

## Slow first start in a new container

**Symptom.** Every new container takes minutes to report ready, while a second start in the same container is fast.

**Cause.** A new container has no compiled kernels. Triton, Inductor, CuTe and JIT CUDA kernels are compiled the first time a stage builds its model and captures its CUDA graphs, and are cached on disk. A container that starts with an empty cache compiles them again on every start.

**Fix.** Build the cache once and ship it in the image:

1. Start the server in a container of the serving image, on the GPU model you deploy, and wait until `/health` returns HTTP 200. Send one request for each configuration you serve.
2. Copy the cache directories into the image at the same paths.

The caches live under `~/.cache/sglang`, set by `SGLANG_CACHE_DIR`. JIT kernels are in `~/.cache/sglang/jit`, set by `SGLANG_JIT_CACHE_DIR`, which does not follow `SGLANG_CACHE_DIR`: set both when you move the root. `TORCHINDUCTOR_CACHE_DIR` defaults to `~/.cache/sglang-omni/torchinductor`; copy that directory too when it is not empty.

For Qwen3-TTS 1.7B CustomVoice on one H100 with `SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0`, `/health` returned HTTP 200 a median of 235 s after `sgl-omni serve` started with an empty cache, and 99 s with the cache in the image (6 new containers each). The cache was 110 MB in 3,503 files.

- Keep the cache in the image or on local disk. It is thousands of small files, and on a network volume each first open is a round trip.
- A cache is reused only where it was built for: the same GPU architecture, driver and installed packages. Rebuild it when the image changes; a miss compiles as before.
- A configuration the first start did not cover compiles on its first request as before.

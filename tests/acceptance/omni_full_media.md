# Official-size Thinker media validation

`omni_full_media.py` adds representative image, audio, and video forward/backward
comparisons to the separate full text entrypoint, `omni_full_size.py`. It uses
the full official dimensions from Qwen/Qwen3-Omni-30B-A3B-Instruct revision
`26291f793822fb6be9555850f06dfe95f2d7e695`; the raw official config must have
SHA256 `eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d`.

All weights are explicitly random. Each named parameter is initialized from a
fixed name-derived seed, and actual weight-byte SHA256 fingerprints must agree
between HF and native execution. This does not validate the released weights.
The native path uses each implemented model's sequential `reference_forward`,
with its full encoder, projections/DeepStack, text decoder, and language head.
It does not exercise the staged distributed scheduler.

| Case | Full parameters | Gradient tensor keys | BF16 weights + gradients | Minimum free GPU memory |
| --- | ---: | ---: | ---: | ---: |
| image or video + text | 31,071,278,320 | 882 | 115.75 GiB | 127.75 GiB |
| audio + text | 31,180,574,080 | 1056 | 116.16 GiB | 128.16 GiB |

The minimum adds 12 GiB for activations and temporary buffers; it is an admission
check, **not a measured peak or guarantee against OOM**. Use one dedicated H200
and 256 GiB host RAM. Both `nvidia-smi` and `torch.cuda.mem_get_info` are recorded.
HF runs first; all reference gradients move to CPU, HF GPU storage is released,
and native weights are reconstructed for the comparison. Two full models never
reside on the GPU together.

The input is the fixed first real sample returned by each existing acceptance
fixture. Processing settings and media are retained. There is no fallback to
another sample or truncation on failure. The declared token cap is 512; the
existing bundle produces image `coco-57870` (14 tokens, 8 patches), audio
`librispeech-374-180298-0000` (247 tokens, 1453 feature frames), and video
`msvd-WTf5EgVY5uU_98_104` (386 tokens, 1512 temporal patches). Sample identity,
shapes, positions, processor revision, and bundle checksum are saved.

Use the reviewed GPU image with torch `2.13.0+cu130`, Transformers `5.17.0`,
and the existing pinned offline processor cache. Set `HF_HUB_OFFLINE=1`.
The executable checks the exact target commit and clean tracked files before
and after the run. Supply the final integrated commit, not this document's
creation commit.

```bash
python -m tests.acceptance.omni_full_media \
  --official-config /artifacts/config.json --expected-commit <40-hex-final-commit> \
  --dataset /artifacts/data --modality image --weight-source random --plan

timeout --signal=TERM --kill-after=10s 600s \
  python -m tests.acceptance.omni_full_media \
  --official-config /artifacts/config.json --expected-commit <40-hex-final-commit> \
  --dataset /artifacts/data --modality image --weight-source random \
  --execute --out /evidence/full-image
```

Run `audio` and `video` sequentially in separate processes with separate fresh
output directories and the same 600-second outer timeout. A Modal controller
must independently bound its allocation and concurrency, account for all costs,
retain process logs, and classify timeout/termination. This entrypoint makes
no cloud allocation, billing, or download calls.

Logits require normalized squared error below `1e-3`; cross entropy uses
`rtol=atol=1e-3`. Every parameter gradient and media-input gradient requires
normalized squared error below `1e-2`. Missing gradients fail. Tiny and exact
zero gradients are compared too, with every active component required to
contribute a nonzero gradient. All parameter rows include finite status,
norms, maximum absolute values, maximum absolute differences, and NSE before
the aggregate numeric verdict, including on a numeric failure.

`summary.json` reports HF/native forward and backward separately, measured
peaks and time, source, full configuration, dependencies, and weight origin.
FP32 `main_grad`, Adam update, checkpoint restore, distributed topologies, and
released-weight validation are not performed by this gate. In particular,
full FP32 main gradients and resident Adam state exceed one H200; an accepted
BF16 forward/backward result must not be described as full optimizer acceptance.

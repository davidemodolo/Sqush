# Baking

**Baking** turns a freshly downloaded model into a compact *cooked* checkpoint on disk, then deletes the bulky raw download. Serving always loads the cooked model — smaller and faster to start (no on‑the‑fly re‑quantization).

Cooked path convention (`_cooked_model_path`): the raw directory with a `-cooked` suffix, e.g. `models/Qwen__Qwen3.6-27B` → `models/Qwen__Qwen3.6-27B-cooked`.

## When it happens

- `bake` runs it explicitly.
- `serve`/`chat` check for the cooked model first and bake on first run if it's missing.

Both compute the cooked path from config **before** downloading, so a second run doesn't re‑download tens of GB before discovering the cooked model already exists.

## Two strategies (`_bake_model` dispatches by tier)

### LOW tier — CPU side‑car bake (`_bake_safetensors`)

The source is already a bitsandbytes 4‑bit repo, so the LM weights are packed 4‑bit on disk. The bake is a **pure‑CPU, tensor‑by‑tensor** pass (GPU peak < 100 MB) that:

1. Copies all non‑shard files (config, tokenizer, processor).
2. Patches `config.json`: removes the visual‑encoder module names from `llm_int8_skip_modules` so bitsandbytes quantizes the vision encoder at load. **`lm_head` stays in the skip list** — on a pre‑quantized checkpoint, `from_pretrained` expects `Linear4bit` weights to already be packed; leaving `lm_head` out would load raw bf16 into a `Linear4bit` with no `quant_state` and crash on the first forward. `lm_head` is instead NF4‑quantized post‑load.
3. Quantizes `embed_tokens` to a 4‑bit per‑group asymmetric **side‑car** (`quantized_embeddings.safetensors`) and writes a placeholder into the shard so `from_pretrained` never allocates the 1.93 GB bf16 table. Sets `qs_pre_baked_embeddings = true` in the config.

### HIGH tier — GPU NF4 bake (`bake_nf4_checkpoint`)

The source is full bf16 (~52 GB), so the LM weights must be **genuinely** NF4‑quantized — and bitsandbytes requires CUDA for that. The bake:

1. Loads the model with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` on a **pure‑GPU** `device_map="cuda:0"` (serialization can't use CPU offload).
2. `model.save_pretrained(cooked, safe_serialization=True)` — writes the packed NF4 shards plus `quantization_config`.
3. Saves the tokenizer and `Qwen3VLProcessor` alongside.

The vision encoder is a set of `Linear` layers, so bitsandbytes quantizes it too — it is preserved in the cooked checkpoint (4‑bit), exactly as the old on‑the‑fly HIGH path had it. `embed_tokens` (an `nn.Embedding`) stays bf16.

!!! info "Validated on real hardware"
    On an RTX 3090 the 27B bake takes ~50 s, produces a 17.9 GB cooked checkpoint (from 52 GB), reloads via the pre‑quantized fast path at 17.9 GB peak, and generates coherent text.

## Transient disk usage

During a bake the raw and cooked models coexist:

- 27B: ~52 GB raw + ~18 GB cooked ≈ **~70 GB** peak, settling to ~18 GB after the raw is deleted.

The raw is only deleted **after** the bake succeeds — if `bake_nf4_checkpoint` raises, the raw is left intact.

## Detection on reload

`_model_is_pre_quantized` checks `config.json` for `quantization_config.quant_method == "bitsandbytes"`. A cooked HIGH checkpoint matches, so `load_and_quantize_model` takes the fast **already‑quantized** branch (logging `Loaded pre-quantized bitsandbytes model from disk`) instead of re‑quantizing bf16.

## Download completeness

`download.py`'s `_download_complete()` guards against treating an interrupted download as finished. A directory counts as complete only when its `model.safetensors.index.json` is present **and every shard it references exists** (or, unsharded, `model.safetensors` + `config.json`). Otherwise the download falls through to `snapshot_download(resume_download=True)`.

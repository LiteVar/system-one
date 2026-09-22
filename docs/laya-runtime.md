# Laya Multilingual native runtime

## Assets

The initial model is `Weidows/laya-multilingual-GGUF`:

- `laya-multilingual-Q8_0.gguf`: ModernBERT/mmBERT encoder;
- `laya-multilingual-head.safetensors`: the separate Laya decision head;
- `tokenizer.json`: the Hugging Face tokenizer, which is authoritative for token IDs.

The safetensors metadata contains the Laya runtime configuration. The model card reports a 22-layer, 768-wide encoder and recommends Q8 for decision-sensitive use.

## Reference sequence

For one question, the original Laya implementation builds:

```text
[CLS] <type> question: <instructions> [SEP]
[MASK] <option 0> [MASK] <option 1> ... [SEP]
<serialized state> [SEP]
```

The marker positions are the positions of the `[MASK]` tokens. User instructions and state text have literal mask tokens replaced with spaces before tokenization. Options are prefixed with one space and truncated to 48 tokens before the shared head budget is applied.

The multilingual configuration uses `max_len=1024` and `head_max_len=256`.

Question types are encoded as `choice=0`, `score=1`, and `noul=2`. Noul options are always ordered `[false, true]`.

## Head

The native head performs:

1. encoder hidden states from llama.cpp;
2. add question-type embedding to every token;
3. two pre-norm Transformer encoder layers with the reference ReLU feed-forward
   activation;
4. gather the hidden state at each option marker;
5. score each marker with the scorer MLP;
6. apply a question-specific softmax and optional temperature;
7. map the distribution to Choice, Score, or Noul.

The action head receives the unnormalized first-token representation plus the
top-probability, top-two margin, entropy, and option-count features. Its first output
is exposed as Laya's `action.act_probability` field.

Choice and Score confidence is normalized Shannon entropy confidence:

```text
1 - H(p) / log(number_of_options)
```

Score is the probability-weighted level index. Noul is the probability at the `true` marker.

## llama.cpp contract

System One uses the native C API, not `llama-server`:

- load the GGUF as an encoder-only model;
- use non-causal attention;
- set embedding pooling to `none`;
- call `llama_encode`;
- read each token vector with `llama_get_embeddings_ith`.

The runtime loader validates the required C symbols and version metadata before a
runtime is selected. Loading a Laya model verifies the expected 768-wide output
dimension, and the first actual request exercises `llama_encode` and per-token
embedding extraction. This is intentional: some compatible ModernBERT GGUF builds
do not advertise the encoder through a dedicated capability flag even though the
non-causal encode path works. A runtime is not accepted merely because a file named
`libllama` exists.

## Compatibility and calibration

The model ships with neutral temperatures in its configuration. The runtime preserves those values and exposes the calibration status in diagnostics. It does not claim that raw Laya probabilities are universally calibrated.

The dynamic C ABI is pinned to the llama.cpp header contract used by this project. The
managed runtime is pinned to the `b10964` upstream release, carries a manifest with
target/backend/checksum metadata, and is revalidated through symbol loading before
use. External runtimes are supported but must expose the same encoder/embedding C API.

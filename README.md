# llmash-ablate

`llama-ablate`, the tool behind [`llmash ablate`](https://github.com/omgitsbase/llmash#abliteration).
It removes a chat model's refusals without touching its weights: it finds the direction each
layer's residual stream uses for "refuse", searches for how much of it to take out, and writes a
small control vector that projects it away at inference. The model file is never changed, and the
vector works on any quantization of it.

| model | refusals before | after | first-token KL | search |
|---|--:|--:|--:|--:|
| Qwen3.8-27B GSQ-RCO | 94/100 | 2/100 | 0.113 | 70 s |
| Qwen3.5-4B NVFP4 | 98/100 | 22/100 | 0.098 | 21 s |
| Gemma 4 E4B Q4_K_M | 98/100 | 3/100 | 0.058 | 14 s |
| Gemma 4 26B-A4B RCO-3 | 99/100 | 2/100 | 0.196 | 17 s |

Refusals are counted on 100 held-out prompts from `mlabonne/harmful_behaviors` by a model reading
each whole answer, not by matching words. KL is how far the first answer token moves on harmless
prompts. Times are on one RTX PRO 6000.

## Method

Directional ablation (Arditi et al. 2024), searched the way Heretic does it:

1. The direction at each layer is the mean residual at the last prompt token over prompts the model
   refuses, minus the mean over prompts it answers, with the answered mean's own component removed
   so ordinary prompts keep their bearing. Each layer gets its own direction.
2. A per-layer weight kernel says how much of that component each layer loses.
3. Each candidate is scored on held-out prompts: how many harmful prompts still draw a refusal, and
   the KL divergence of the first answer token on harmless prompts against the untouched model. The
   search keeps the candidate with the fewest refusals at a low KL, and writes nothing when the best
   one still lands above `--max-kl` (1.0 by default).
4. The output is a control-vector GGUF with `controlvector.mode = "project"`: per layer, the square
   root of the weight times the unit direction, applied as `cur - v (v . cur)`.

Each candidate's prompts run side by side as one batch of sequences (`--seqs`, 128 by default and
halved until the context fits), so a 16-pass search of a 27B takes about a minute.

## Use

The easy way is llmash, which fetches the prompt sets, runs the search on its own runtime, and loads
the vector beside the model from then on:

```
llmash ablate MODEL
```

By hand:

```
llama-ablate -m model.gguf --harmful bad.txt --harmless good.txt \
  --harmful-eval bad_test.txt --harmless-eval good_test.txt \
  --level balanced -o model.ablation.gguf --report report.json
```

Prompt files hold one prompt per line. `--level quick|balanced|thorough|best` sets how long the
search runs. `--judge-url` scores refusals with a chat model served at that URL instead of marker
words. Progress goes to stdout as `ablate: ` lines. Every llama.cpp option, such as `-ngl` or `-c`,
passes through.

The vector loads in llmash's runtime with `--control-vector model.ablation.gguf`. Upstream
llama.cpp would add it to the stream instead of projecting it out, so use the fork.

## Build

It builds against [llmash's llama.cpp fork](https://github.com/omgitsbase/llama.cpp/tree/llmash),
which CMake fetches, or a checkout of it:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release            # add -DGGML_CUDA=ON for an NVIDIA card
cmake -B build -DLLAMA_CPP_DIR=../llama.cpp          # ...or use a checkout you already have
cmake --build build --config Release --target llama-ablate
```

The same source ships as `tools/ablate` in the fork, and llmash's runtime builds it from there.

## License

MIT, like llmash. llama.cpp is MIT, by the ggml authors.

# Hindi-to-English Machine Translation From Scratch

This is a from-scratch encoder-decoder Transformer for Hindi-to-English translation.
It started as a learning project and keeps that constraint deliberately: every model
sublayer is implemented by hand, with no pretrained translation checkpoints and no
fine-tuning of mBART, NLLB, IndicTrans2, or Hugging Face models.

The model uses PyTorch primitives such as `nn.Linear`, `nn.Embedding`, and
`nn.Dropout`, but it does not use `nn.Transformer` or `nn.MultiheadAttention`.

## Results

Trained on the IITB English-Hindi corpus, with about 913k sentence pairs after
cleaning and LaBSE filtering. The active tokenizer is a shared 16k SentencePiece
unigram model. Training runs on a single GPU, usually Colab Pro or Kaggle T4.

| Model | Decode | BLEU | chrF++ | TER |
| --- | --- | ---: | ---: | ---: |
| RNN baseline (`learning-phase/`) | greedy | 5.63 | n/a | n/a |
| Transformer base, sinusoidal | greedy with KV cache | 19.25 | 48.70 | 69.75 |
| Transformer base, sinusoidal | beam, `k=5`, `lp=0.6` | **20.07** | **49.08** | **68.28** |

Best dev NLL is **1.937**, or about **6.9** token perplexity. RoPE retraining
and back-translation are in progress.

## Layout

The repo is an installable Python package. Code lives in `src/nmt/`, while the
notebooks are thin drivers for Colab and Kaggle runs.

```text
hindi-translator/
  learning-phase/      # archived RNN learning phase (frozen baseline, BLEU 5.63)
  src/nmt/             # the package (import as `nmt`)
    config.py          #   ModelConfig / DataConfig / TrainConfig / DecodeConfig
    model/             #   embeddings, positional (sin/learned/RoPE), attention (MHA + KV cache),
                       #     feedforward (ReLU/GeLU/SwiGLU/GeGLU), norm (LN/RMSNorm, pre/post),
                       #     encoder, decoder, transformer
    data/              #   download (IITB), clean (NFC + script-check langid + ratio + dedupe +
                       #     leak), labse_filter, tokenizer (SentencePiece), dataset (token-batch)
    train/             #   optim (AdamW), schedule (inv-sqrt + cosine), loss (label-smoothed NLL),
                       #     ema (EMA + SWA), checkpoint (resumable + averaging), tracking
                       #     (TB/wandb), profiling (torch.profiler), loop (AMP + grad-accum + clip)
    decode/            #   greedy (batched + KV cache), beam (length + coverage penalty), cache,
                       #     translate, plus mbr / ensemble stubs
    eval/              #   metrics (BLEU / chrF++ / TER via sacrebleu), evaluate driver
    analysis/          #   attention viz, head importance, alignments, embeddings, probing
  notebooks/           # thin drivers: 01_data, 02_train (Colab + Kaggle),
                       #   03_backtranslation, 04_decode_eval (Colab + Kaggle),
                       #   05_analysis, 00_run_tests
  tests/               # shape/sanity tests (run via `PYTHONPATH=src pytest`)
  pyproject.toml
```

## Status

- **Model (T2):** the base Transformer path is complete: sinusoidal positions,
  LayerNorm, pre-norm, ReLU FFN, MHA, and three-way tied embeddings. KV cache
  and RoPE are also implemented. ALiBi, MQA, GQA, Shaw relative positions, and
  drop-path are deferred.
- **Data (T1):** download, cleaning, LaBSE filtering, tokenizer training, and
  dataset loading are done. Back-translation is in progress.
- **Train (T3):** AdamW, inverse-square-root and cosine schedules,
  label-smoothed NLL, automatic bf16/fp16 AMP, token-normalized gradient
  accumulation, clipping, EMA, SWA, resumable checkpoints, checkpoint averaging,
  TensorBoard tracking, and profiling are implemented.
- **Decode (T4):** greedy decoding with KV cache and beam search with length and
  coverage penalties are implemented. MBR, ensembling, and diverse beam search
  are still pending.
- **Eval (T6):** BLEU, chrF++, and TER are reported with sacrebleu. Manual
  evaluation is still pending.
- **Analysis (T7):** analysis modules are currently stubs.

## Quickstart

```bash
git clone https://github.com/Rishi-Jain-27/hindi-translator
cd hindi-translator
pip install -e .

# tests (torch needed for most; a few are torch-free)
PYTHONPATH=src pytest

# end-to-end pipeline (run on a GPU box: Colab Pro or Kaggle T4)
#   notebooks/01_data.ipynb           IITB download, clean, LaBSE filter, train tokenizer
#   notebooks/02_train.ipynb          train forward hi->en with AMP, EMA, and resumable checkpoints
#   notebooks/02b_train_reverse.ipynb train reverse en->hi for back-translation
#   notebooks/04_decode_eval.ipynb    translate test set and report BLEU / chrF++ / TER
```

Data is not committed. IITB is downloaded on the GPU runtime, and cached corpus
files and checkpoints live in Google Drive for Colab or Kaggle Datasets for
Kaggle runs.

## Constraints

- **From scratch only.** The translator uses no pretrained weights and no
  fine-tuning. A future speech-to-speech app may use pretrained ASR and TTS
  around the translator, but the translation model itself stays hand-built.
- **Single GPU.** No DDP. The project is designed for Colab Pro and Kaggle T4.
- **No heavy neural metrics.** Evaluation uses BLEU, chrF++, TER, and manual
  review. COMET and BLEURT are intentionally excluded because they compete for
  VRAM during evaluation.

## Notes

The model is also available on Kaggle.

## License and Credits

Educational project.

- Corpus: [IITB parallel corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
- Tokenizer: SentencePiece
- Metrics: sacrebleu
- Pair filtering: LaBSE

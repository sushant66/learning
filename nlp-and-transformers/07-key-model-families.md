# Chapter 7: Key Model Families — The Models That Shaped the Field

## A Family Tree

Every model you hear about today — GPT-4, Claude, LLaMA, Gemini — has ancestors. Understanding the lineage helps you see why certain design choices were made and where the field is heading.

```
  2017  ──── Original Transformer (Google)
                    │
          ┌─────────┼──────────┐
          ↓         ↓          ↓
  2018  BERT     GPT-1        ── (encoder-decoder line)
       (Google)  (OpenAI)          │
          │         │              │
  2019  RoBERTa  GPT-2          T5
       (Meta)    (OpenAI)      (Google)
          │         │              │
  2020    │      GPT-3          mT5, BART
          │      (OpenAI)
          │         │
  2022  DeBERTa    │           FLAN-T5
                   │
  2023         LLaMA 1,2      ── (open-source explosion)
               (Meta)             │
                │            Mistral, Mixtral
                │
  2024      LLaMA 3, 3.1
               (Meta)
```

Let's meet each major family.

---

## BERT (2018) — The Understanding Champion

**Full name:** Bidirectional Encoder Representations from Transformers
**By:** Google AI
**Architecture:** Encoder-only
**Pre-training:** Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)

### What Made BERT Special

Before BERT, most NLP systems trained separate models for each task — one for sentiment analysis, another for question answering, another for named entity recognition. You'd need labeled data for each.

BERT changed the game: **pre-train once, fine-tune for anything.**

```
  Pre-trained BERT (understands language)
          │
    ┌─────┼─────┬──────────┬──────────┐
    ↓     ↓     ↓          ↓          ↓
  + tiny  + tiny + tiny    + tiny     + tiny
  classifier     classifier classifier classifier
    ↓     ↓     ↓          ↓          ↓
  Sentiment  NER  Question  Paraphrase  ...
  Analysis        Answering Detection
```

Add a small classification layer on top, fine-tune on a small labeled dataset, and BERT crushed every NLP benchmark.

### Architecture Details

```
BERT-Base:   12 layers, 768 hidden, 12 heads,  110M parameters
BERT-Large:  24 layers, 1024 hidden, 16 heads, 340M parameters
```

### The Training Objectives

**Masked Language Modeling (MLM):** Randomly mask 15% of tokens, predict them using bidirectional context.

**Next Sentence Prediction (NSP):** Given two sentences, predict if sentence B actually follows sentence A in the original text.

```
Input:  [CLS] The cat sat on the mat [SEP] It was a nice day [SEP]
Label:  IsNextSentence (True)

Input:  [CLS] The cat sat on the mat [SEP] Stocks rose 3% today [SEP]
Label:  NotNextSentence (False)
```

NSP was later shown to be not very useful — RoBERTa dropped it and got better results.

### BERT's Legacy

BERT proved that transformer-based pre-training works. It democratized NLP — suddenly, a small team with a GPU could fine-tune BERT and build a state-of-the-art classifier. You'll still encounter BERT and its descendants in production systems today, especially for search and classification.

### Key Descendants

| Model | Year | Improvement |
|-------|------|-------------|
| **RoBERTa** (Meta) | 2019 | Removed NSP, trained longer, more data |
| **ALBERT** (Google) | 2019 | Parameter sharing to reduce model size |
| **DeBERTa** (Microsoft) | 2020 | Disentangled attention (separate content and position) |
| **ELECTRA** (Google) | 2020 | Replaced MLM with "detect replaced tokens" — more efficient |

---

## GPT Series (2018-2024) — The Generation Revolution

### GPT-1 (2018): The Proof of Concept

**By:** OpenAI
**Architecture:** Decoder-only, 12 layers, 117M parameters
**Pre-training:** Causal Language Modeling (CLM)
**Data:** BookCorpus (~7,000 books)

GPT-1 showed that a decoder-only transformer, pre-trained on next-token prediction, could be fine-tuned for various tasks. It was decent but not spectacular — BERT outperformed it on most benchmarks.

What mattered was the idea: **language modeling as pre-training for everything.**

### GPT-2 (2019): Scale Changes Things

**Parameters:** 1.5B (10× GPT-1)
**Data:** WebText (40GB of high-quality web pages)

GPT-2 demonstrated something surprising: with enough scale, you don't always need fine-tuning. You can get the model to do tasks **zero-shot** — just by describing the task in the prompt.

```
Prompt: "Translate English to French: cheese →"
GPT-2:  "fromage"
```

No translation-specific training. No labeled data. The model figured it out from seeing translated text during pre-training.

OpenAI initially didn't release the full model, calling it "too dangerous." This was the beginning of the public conversation about AI safety.

### GPT-3 (2020): The In-Context Learning Breakthrough

**Parameters:** 175B (100× GPT-2)
**Data:** 300B tokens from Common Crawl, WebText, books, Wikipedia
**Key innovation:** **Few-shot learning via prompting**

GPT-3 was a paradigm shift. It showed that a large enough model can learn new tasks from just a few examples in the prompt:

```
Zero-shot (no examples):
  "Translate to French: Hello → "

One-shot (one example):
  "Translate to French:
   Hello → Bonjour
   Goodbye → "

Few-shot (several examples):
  "Translate to French:
   Hello → Bonjour
   Goodbye → Au revoir
   Thank you → Merci
   How are you? → "
```

GPT-3's few-shot performance was competitive with models specifically fine-tuned for those tasks. This was remarkable — and it meant you could build AI applications without training a model at all.

### GPT-3.5 / ChatGPT (2022): The Public Awakening

ChatGPT wasn't a new base model — it was GPT-3.5 (a refined GPT-3) fine-tuned with:
1. **Supervised Fine-Tuning (SFT)** on conversation data
2. **Reinforcement Learning from Human Feedback (RLHF)** to align with human preferences

This combination turned a next-token predictor into a conversational assistant. It reached 100 million users in 2 months — the fastest-growing consumer application in history.

### GPT-4 (2023): Multimodal and Powerful

Details are less public, but GPT-4 is believed to be a **Mixture of Experts (MoE)** model — multiple smaller expert networks that are selectively activated for different inputs. It also added vision capabilities (understanding images).

```
  Traditional Transformer:
  Every input → ALL parameters active

  Mixture of Experts:
  Every input → Router → activates 2 of 8 experts
                         (only ~25% of parameters active per token)
```

This allows having a very large total model while keeping per-token computation manageable.

---

## T5 (2019) — The Text-to-Text Approach

**Full name:** Text-to-Text Transfer Transformer
**By:** Google Research
**Architecture:** Encoder-decoder
**Key insight:** Frame every NLP task as text generation

T5 treated everything — classification, translation, summarization, question answering — as a text-to-text problem:

```
Task prefix tells the model what to do:

"translate English to German: That is good" → "Das ist gut"
"summarize: [long text]"                     → "summary text"
"cola sentence: The course is jumping well"  → "not acceptable"
"stsb sentence1: The cat sat. sentence2: A cat sitting." → "4.2"
```

### The T5 Study

The T5 paper was partly a massive ablation study — they systematically tested dozens of choices:
- Pre-training objectives (MLM vs CLM vs span corruption)
- Model sizes (60M to 11B)
- Training data composition
- Fine-tuning strategies

Their conclusion: **span corruption** (masking contiguous spans) + **encoder-decoder** architecture + **large scale** = best results on benchmarks.

### Variants

| Model | What Changed |
|-------|-------------|
| **mT5** | Multilingual version, trained on 101 languages |
| **FLAN-T5** | Instruction-tuned T5, much better at following prompts |
| **UL2** | Combined multiple pre-training objectives |

---

## LLaMA (2023-2024) — The Open-Source Catalyst

**By:** Meta AI
**Architecture:** Decoder-only
**Key impact:** Made high-quality LLMs available to everyone

### LLaMA 1 (February 2023)

Meta trained a family of models (7B, 13B, 33B, 65B parameters) on **1 trillion tokens** of publicly available data. The key finding: **smaller models trained on more data** can match or beat larger models trained on less data.

```
LLaMA 13B (13 billion parameters)
  outperformed
GPT-3 (175 billion parameters)
  on most benchmarks

How? 10× more training data per parameter.
```

This validated the **Chinchilla scaling laws** (more on this in Chapter 8).

The model weights were leaked online, sparking an explosion of open-source LLM development.

### LLaMA 2 (July 2023)

Officially open-source (with a license). Trained on **2 trillion tokens** — double LLaMA 1.

Key architectural choices:
- **RoPE** (Rotary Position Embeddings) for position encoding
- **GQA** (Grouped-Query Attention) for efficient inference
- **SwiGLU** activation in FFN
- **RMSNorm** instead of LayerNorm
- **4K context window** (extended to 32K in Code Llama)

```
LLaMA 2 Architecture Improvements:

Standard Multi-Head Attention       Grouped-Query Attention (GQA)
┌────┐ ┌────┐ ┌────┐ ┌────┐       ┌────┐ ┌────┐ ┌────┐ ┌────┐
│ Q₁ │ │ Q₂ │ │ Q₃ │ │ Q₄ │       │ Q₁ │ │ Q₂ │ │ Q₃ │ │ Q₄ │
└──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘       └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘
   │      │      │      │             └──┬───┘      └──┬───┘
┌──▼─┐ ┌──▼─┐ ┌──▼─┐ ┌──▼─┐          ┌──▼──┐       ┌──▼──┐
│ K₁ │ │ K₂ │ │ K₃ │ │ K₄ │          │ K₁₂ │       │ K₃₄ │
│ V₁ │ │ V₂ │ │ V₃ │ │ V₄ │          │ V₁₂ │       │ V₃₄ │
└────┘ └────┘ └────┘ └────┘          └─────┘       └─────┘

4 KV heads (more memory)            2 KV heads (less memory, faster)
```

GQA shares Key-Value heads across multiple Query heads, dramatically reducing KV-cache memory and speeding up inference.

### LLaMA 3 / 3.1 (2024)

- Trained on **15 trillion tokens** — a massive jump
- 8B, 70B, and 405B parameter versions
- **128K context window**
- Added multilingual support and tool use capabilities
- The 405B model is competitive with GPT-4

### The LLaMA Effect

LLaMA's open release triggered an ecosystem:

```
LLaMA → Alpaca (Stanford, instruction-tuned)
      → Vicuna (LMSYS, chat-tuned)
      → Code Llama (Meta, code-specialized)
      → Mistral, Mixtral (Mistral AI, architectural innovations)
      → Many fine-tuned variants on HuggingFace
```

---

## Mistral & Mixtral (2023-2024) — Efficiency Innovators

**By:** Mistral AI (France)

### Mistral 7B

A 7B parameter model that outperformed LLaMA 2 13B. Key innovations:

- **Sliding Window Attention (SWA):** Instead of attending to all previous tokens, each layer attends to a fixed window (e.g., 4096 tokens). By stacking layers, information can still propagate across the full sequence.

```
Standard Causal Attention (attends to everything before):
Token 5000 attends to → tokens 1 through 4999 (expensive!)

Sliding Window Attention (window = 4096):
Token 5000 attends to → tokens 904 through 4999 (fixed cost)

But with 32 layers stacked:
Information from token 1 can reach token 5000
through intermediate tokens across layers
```

### Mixtral 8x7B — Mixture of Experts

A sparse MoE model with 8 expert FFN networks per layer. A router selects the top 2 experts for each token:

```
                     Input token
                         │
                    ┌────▼────┐
                    │  Router  │ (small neural network)
                    └────┬────┘
                         │
          Scores: [0.1, 0.05, 0.4, 0.02, 0.3, 0.03, 0.05, 0.05]
                                ↑                ↑
                           Expert 3          Expert 5
                          (selected)         (selected)
                              │                  │
                              ▼                  ▼
                         ┌────────┐         ┌────────┐
                         │Expert 3│         │Expert 5│
                         │  FFN   │         │  FFN   │
                         └───┬────┘         └───┬────┘
                             │                  │
                             ▼                  ▼
                      Weighted combination → Output
```

Total parameters: 46.7B. Active parameters per token: ~12.9B. This gives you the quality of a much larger model with the inference cost of a smaller one.

---

## The Evolution at a Glance

```
  2018    2019    2020    2021    2022    2023    2024
   │       │       │       │       │       │       │
   │  BERT │       │       │       │       │       │
   │  110M │       │       │       │       │       │
   │       │       │       │       │       │       │
   │  GPT-1│  GPT-2│  GPT-3│       │ChatGPT│  GPT-4│
   │  117M │  1.5B │  175B │       │       │~1.7T? │
   │       │       │       │       │       │       │
   │       │   T5  │       │       │FLAN-T5│       │
   │       │  11B  │       │       │       │       │
   │       │       │       │       │       │       │
   │       │       │       │       │ LLaMA │LLaMA 3│
   │       │       │       │       │  65B  │  405B │
   │       │       │       │       │       │       │
   │       │       │       │       │       │Mixtral│
   │       │       │       │       │       │ 8x7B  │
```

The trend is clear: models are getting bigger, trained on more data, and architecturally smarter about how they use their parameters.

---

## How to Choose a Model (Practical Guide)

| Use Case | Recommended | Why |
|----------|------------|-----|
| Build a chatbot | Claude, GPT-4, LLaMA 3 | Best at conversation and instruction following |
| Semantic search / embeddings | BERT-based (e.g., `all-MiniLM`) or specialized embedding models | Encoder models produce better embeddings |
| Text classification | BERT / DeBERTa (fine-tuned) or LLM with prompting | Depends on scale and budget |
| Self-hosted / private | LLaMA 3, Mistral | Open weights, run on your own hardware |
| Cost-sensitive API use | Claude Haiku, GPT-4o-mini, Mistral Small | Smaller, cheaper models for simpler tasks |
| Code generation | Claude, GPT-4, Code Llama | Strong code understanding |

---

## Key Takeaways

1. **BERT** proved pre-training works for understanding. Still relevant for embeddings and classification.
2. **GPT** showed that decoder-only + scale = general-purpose AI. Each generation was a step change.
3. **T5** unified all tasks into text-to-text. Its ideas influence how we prompt models today.
4. **LLaMA** democratized LLMs. Open-source models are now competitive with proprietary ones.
5. **Mistral/Mixtral** proved that architectural innovations (SWA, MoE) can punch above their weight class.
6. **The trend is toward decoder-only, open-source, and mixture-of-experts.**

---

## What's Next?

We've seen models grow from 110M to hundreds of billions of parameters. But is bigger always better? What's the relationship between model size, training data, and compute? And why do certain abilities seem to appear suddenly at certain scales?

The answers lie in scaling laws and emergent abilities — the fascinating (and sometimes unsettling) science of what happens as models grow.

---

**Next: [Scaling & Emergent Abilities — Why Bigger Models Get Smarter](./08-scaling-and-emergent-abilities.md)**

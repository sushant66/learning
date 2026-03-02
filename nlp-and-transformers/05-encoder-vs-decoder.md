# Chapter 5: Encoder vs Decoder — Three Flavors of Transformers

## One Architecture, Three Designs

The transformer architecture is like a Lego set — you can assemble the same blocks in different ways for different purposes. Over time, three distinct designs emerged, each optimized for different tasks.

Understanding these three flavors is essential because it explains **why** GPT, BERT, and T5 behave so differently, even though they're all built from the same components.

---

## Flavor 1: Encoder-Only (BERT)

### The Design

An encoder-only transformer reads the entire input at once and produces a rich representation for each token. There's no generation step — it doesn't produce new text.

```
  Input: "The cat [MASK] on the mat"

  ┌──────────────────────────────────┐
  │        Encoder Stack (×N)        │
  │                                  │
  │  Every token can see ←→ every    │
  │  other token (bidirectional)     │
  │                                  │
  └──────────────┬───────────────────┘
                 │
                 ▼
  Contextualized representations for each token
         ↓         ↓        ↓
     classify   fill mask   extract info
```

### The Key Feature: Bidirectional Attention

In an encoder, every token can attend to every other token — both to the left AND to the right. There's no causal mask.

```
                 Attending to:
                "The"  "cat"  "[MASK]"  "on"  "the"  "mat"
  "The"   sees: [  ✓     ✓       ✓       ✓     ✓      ✓  ]
  "cat"   sees: [  ✓     ✓       ✓       ✓     ✓      ✓  ]
  "[MASK]" sees:[  ✓     ✓       ✓       ✓     ✓      ✓  ]
  "on"    sees: [  ✓     ✓       ✓       ✓     ✓      ✓  ]
  "the"   sees: [  ✓     ✓       ✓       ✓     ✓      ✓  ]
  "mat"   sees: [  ✓     ✓       ✓       ✓     ✓      ✓  ]

  Full bidirectional attention — every token sees everything
```

This is incredibly powerful for **understanding** text. To predict that [MASK] should be "sat", the model can use both "cat" (before) and "on the mat" (after). It has the complete picture.

### Examples: BERT, RoBERTa, ELECTRA, DeBERTa

### Best For

- **Text classification**: Is this email spam or not? Is this review positive or negative?
- **Named entity recognition**: Find all person names, locations, dates in this text.
- **Extractive question answering**: Given a passage, find the span that answers the question.
- **Sentence embeddings**: Create a vector representation of a sentence for search or similarity.

### The Limitation

Encoder-only models can't generate text. They can fill in blanks and classify, but they can't write a paragraph, translate a sentence, or have a conversation. For that, you need a decoder.

---

## Flavor 2: Decoder-Only (GPT)

### The Design

A decoder-only transformer generates text one token at a time. Each token can only attend to previous tokens (causal/autoregressive).

```
  Input: "The capital of France is"

  ┌──────────────────────────────────┐
  │        Decoder Stack (×N)        │
  │                                  │
  │  Each token can only see ←       │
  │  tokens before it (causal)       │
  │                                  │
  └──────────────┬───────────────────┘
                 │
                 ▼
  Next token prediction → "Paris"
```

### The Key Feature: Causal (Masked) Attention

```
                 Attending to:
                "The"  "capital"  "of"  "France"  "is"
  "The"   sees: [  ✓      ✗        ✗      ✗       ✗  ]
  "capital"sees:[  ✓      ✓        ✗      ✗       ✗  ]
  "of"    sees: [  ✓      ✓        ✓      ✗       ✗  ]
  "France" sees:[  ✓      ✓        ✓      ✓       ✗  ]
  "is"    sees: [  ✓      ✓        ✓      ✓       ✓  ]

  Lower-triangular mask — each token only sees the past
```

### Examples: GPT-2, GPT-3, GPT-4, Claude, LLaMA, Mistral, Falcon

### Best For

- **Text generation**: Write stories, code, emails, anything.
- **Conversational AI**: Chatbots and assistants.
- **Few-shot learning**: Give it a few examples in the prompt and it learns the pattern.
- **General-purpose tasks**: With the right prompt, a decoder can do classification, translation, summarization — basically anything.

### Why Decoder-Only Won

This is the dominant architecture today, and there are several reasons:

**1. Simplicity.** One architecture, one training objective (predict the next token). No special [MASK] tokens, no separate pre-training tasks.

**2. Scaling.** Decoder-only models scale better with more parameters and data. The Chinchilla paper showed that this architecture efficiently converts compute into capability.

**3. Generality.** A decoder can be prompted to do any task:

```
Classification:  "Is this spam? Email: 'Win $1000!' → Answer: Yes"
Translation:     "Translate to French: 'Hello' → Bonjour"
Summarization:   "Summarize: [long text] → Summary:"
Code:            "Write a Python function that..."
```

You don't need different model architectures for different tasks — one model does everything.

**4. In-context learning.** Decoder models can learn new tasks from examples provided in the prompt, without any weight updates. This emergent ability doesn't appear in encoder-only models.

---

## Flavor 3: Encoder-Decoder (T5)

### The Design

The original transformer design. An encoder reads the full input, a decoder generates the output. They're connected by cross-attention.

```
  Input: "Translate English to French: The cat is on the mat"

  ┌────────────────┐         ┌────────────────┐
  │    Encoder     │         │    Decoder     │
  │   (×N blocks)  │         │   (×N blocks)  │
  │                │         │                │
  │  Bidirectional │ ──────→ │  Causal        │
  │  self-attention│  cross  │  self-attention │
  │                │  attn   │  +             │
  │                │         │  cross-attention│
  └────────────────┘         └────────────────┘
                                     │
                                     ▼
                             "Le chat est sur le tapis"
```

### How Cross-Attention Works

The decoder block has an extra layer compared to a decoder-only transformer:

```
  Decoder Block:
  ┌────────────────────────────────┐
  │  1. Masked Self-Attention       │  ← decoder attends to its own outputs
  │  2. Cross-Attention             │  ← decoder attends to encoder outputs
  │  3. Feed-Forward Network        │
  └────────────────────────────────┘
```

In cross-attention:
- **Queries** come from the decoder (what the decoder is looking for)
- **Keys and Values** come from the encoder (the input representation)

This lets the decoder "look at" the input at every step of generation.

### Examples: T5, BART, mBART, FLAN-T5

### Best For

- **Translation**: Read the full source, generate the target.
- **Summarization**: Read the full document, generate a summary.
- **Any task where input and output are clearly separate.**

### Why It Lost (For General-Purpose LLMs)

Encoder-decoder models are excellent at structured input→output tasks. But:
- They require separate encoder and decoder, roughly doubling architecture complexity.
- Scaling both halves is less efficient than scaling one decoder.
- For general-purpose chat/generation, the two-stage setup doesn't offer much benefit over a single decoder that can attend to the prompt.

That said, encoder-decoder models are still used in specialized applications, particularly for translation and structured generation tasks.

---

## Side-by-Side Comparison

```
                  Encoder-Only          Decoder-Only         Encoder-Decoder
                  ┌─────────┐          ┌─────────┐          ┌──────┬──────┐
                  │ Encoder │          │ Decoder │          │Encode│Decode│
                  │ ←────→  │          │ ←────   │          │←───→ │←──── │
                  └─────────┘          └─────────┘          └──────┴──────┘

Attention:        Bidirectional        Causal (left only)   Both

Sees future?      Yes                  No                   Encoder: yes
                                                            Decoder: no

Generates text?   No                   Yes                  Yes

Pre-training:     Masked LM (MLM)      Next token (CLM)     Span corruption

Best at:          Understanding         Generation           Structured tasks
                  Classification        Chat                 Translation
                  Embeddings            General purpose      Summarization

Key models:       BERT, RoBERTa        GPT, Claude, LLaMA   T5, BART

Status in 2024+:  Niche use            Dominant              Specialized use
```

---

## A Thought Experiment

Why can a decoder-only model (GPT, Claude) do classification even though it's designed for generation?

Because you can frame any task as text generation:

```
Encoder approach (BERT):
  Input:  "I love this movie"
  Output: [positive]  (classification head on top of encoder)

Decoder approach (GPT):
  Input:  "Classify the sentiment: 'I love this movie'. Sentiment:"
  Output: "positive"  (generated as text)
```

The decoder approach is less efficient for pure classification (you're generating tokens instead of just classifying), but it's infinitely more flexible. One model, any task, just by changing the prompt.

This flexibility is why the industry converged on decoder-only transformers for general-purpose AI.

---

## When to Use What (Practical Guide)

| Task | Best Architecture | Why |
|------|------------------|-----|
| Build a chatbot | Decoder-only | Needs to generate conversational text |
| Semantic search | Encoder-only | Need dense embeddings for similarity |
| Spam classifier | Encoder-only (or decoder) | Encoder is more efficient; decoder works too |
| Translation system | Encoder-decoder (or decoder) | Structured input→output mapping |
| Code generation | Decoder-only | Open-ended generation |
| Summarization | Any of the three | Depends on scale and requirements |

---

## Key Takeaways

1. **Encoder-only** (BERT): Bidirectional attention, great for understanding, can't generate.
2. **Decoder-only** (GPT): Causal attention, great for generation, can also do understanding tasks via prompting.
3. **Encoder-decoder** (T5): Both, great for structured input→output, but more complex.
4. **Decoder-only won** because it's simpler, scales better, and can do (almost) everything the others can through prompting.
5. Encoders are still valuable for embeddings and search — you'll encounter them when building RAG systems.

---

## What's Next?

We've been talking about model architectures, but we haven't addressed the most important question: **how do these models learn?**

You can't hand-label billions of examples. Instead, these models learn from raw text using clever training objectives that don't require any human labels at all. That's the magic of pre-training.

---

**Next: [Pre-training Objectives — How Models Learn Language](./06-pre-training-objectives.md)**

# Chapter 4: The Transformer Architecture — Putting It All Together

## The Paper That Started It All

In June 2017, eight researchers at Google published "Attention Is All You Need." The title was bold — almost arrogant. They were claiming that the dominant approach to sequence modeling (RNNs, LSTMs, convolutions) could be replaced entirely by attention mechanisms.

They were right.

The architecture they described — the **Transformer** — became the foundation for BERT, GPT, T5, LLaMA, Claude, and every major language model that followed. Understanding it is understanding the blueprint of modern AI.

Let's build it up, piece by piece.

---

## The Big Picture

At the highest level, the original transformer has two halves:

```
  ┌────────────────────────────────────────────────────┐
  │                                                     │
  │    Input                              Output        │
  │   (source)                           (target)       │
  │      │                                  ↑           │
  │      ▼                                  │           │
  │  ┌────────┐                       ┌──────────┐      │
  │  │Encoder │ ─── context ────────→ │ Decoder  │      │
  │  │(stack) │     (cross-attention)  │ (stack)  │      │
  │  └────────┘                       └──────────┘      │
  │                                                     │
  │  "Understand the input"      "Generate the output"  │
  │                                                     │
  └────────────────────────────────────────────────────┘
```

The **encoder** reads the input and builds a rich representation. The **decoder** generates the output, one token at a time, using both its own previous outputs and the encoder's representation.

But here's the twist: modern LLMs (GPT, Claude, LLaMA) use **only the decoder half**. They threw away the encoder entirely. We'll discuss why in Chapter 5. For now, let's understand both.

---

## Before the Blocks: Input Processing

Before tokens enter the transformer blocks, two things happen:

### Token Embeddings

Each token ID is converted to a dense vector by looking it up in an embedding table.

```
Token ID 3947 ("Token") → [0.12, -0.34, 0.56, 0.78, ...]  (768 or more dimensions)
```

This table is a learned parameter — it starts random and improves during training.

### Positional Encoding

Self-attention has no concept of order. It processes all tokens in parallel, so without positional information, "The cat sat" and "sat cat The" would look identical.

**Positional encoding** adds position information to each token's embedding.

The original paper used **sinusoidal** functions — fixed mathematical patterns that encode position:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This creates a unique pattern for each position that the model can use to determine relative distances between tokens.

```
Position 0: [0.00, 1.00, 0.00, 1.00, ...]
Position 1: [0.84, 0.54, 0.01, 1.00, ...]
Position 2: [0.91, -0.42, 0.02, 1.00, ...]
                ↑
         Each position has a unique "fingerprint"
```

**Modern models use different approaches:**

- **Learned positional embeddings** (GPT-2): The model learns a position vector for each position during training.
- **RoPE — Rotary Position Embeddings** (LLaMA, Mistral): Encodes relative positions by rotating the query and key vectors. This is the current standard because it generalizes better to longer sequences than the model was trained on.
- **ALiBi** (BLOOM): Adds a linear bias to attention scores based on distance.

The final input to the transformer is:

```
  Input = Token Embedding + Positional Encoding
```

---

## The Transformer Block

The transformer is built by stacking identical blocks. Each block has two main components:

```
  ┌─────────────────────────────────────────┐
  │           Transformer Block              │
  │                                          │
  │   Input                                  │
  │     │                                    │
  │     ├──────────────────┐                 │
  │     ▼                  │                 │
  │  Layer Norm            │                 │
  │     │                  │                 │
  │     ▼                  │                 │
  │  Multi-Head            │ (residual       │
  │  Self-Attention        │  connection)    │
  │     │                  │                 │
  │     ▼                  │                 │
  │    Add ←───────────────┘                 │
  │     │                                    │
  │     ├──────────────────┐                 │
  │     ▼                  │                 │
  │  Layer Norm            │                 │
  │     │                  │                 │
  │     ▼                  │                 │
  │  Feed-Forward          │ (residual       │
  │  Network               │  connection)    │
  │     │                  │                 │
  │     ▼                  │                 │
  │    Add ←───────────────┘                 │
  │     │                                    │
  │   Output                                 │
  │                                          │
  └─────────────────────────────────────────┘
```

Let's understand each component.

---

### Layer Normalization

Before each sub-layer (attention or FFN), the input is normalized.

**Why?** Neural networks are sensitive to the scale of their inputs. Without normalization, values can drift to extreme ranges as they pass through many layers, making training unstable.

Layer normalization computes the mean and variance across the features of each token and normalizes:

```
LayerNorm(x) = γ × (x - mean) / √(variance + ε) + β
```

Where γ and β are learnable parameters that allow the model to undo the normalization if needed.

**Pre-norm vs Post-norm:**

The original paper put layer norm **after** the attention/FFN (post-norm). Modern models put it **before** (pre-norm). Pre-norm makes training more stable, especially for very deep networks.

```
Post-norm (original):  x → Attention → Add → LayerNorm
Pre-norm (modern):     x → LayerNorm → Attention → Add
```

Nearly all modern LLMs use pre-norm. Some use **RMSNorm** (Root Mean Square Normalization), which is simpler and faster — it skips the mean subtraction and just divides by the RMS.

---

### The Feed-Forward Network (FFN)

After attention, each token passes through a feed-forward network. This is a simple two-layer neural network applied independently to each token:

```
FFN(x) = Activation(x × W₁ + b₁) × W₂ + b₂
```

```
  Input (d_model = 768)
       │
       ▼
  ┌──────────┐
  │ Linear 1 │  Project up: 768 → 3072 (4× expansion)
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │Activation│  Non-linearity (GeLU or SiLU)
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ Linear 2 │  Project back down: 3072 → 768
  └────┬─────┘
       │
       ▼
  Output (d_model = 768)
```

**Key details:**

- The inner dimension is typically **4× the model dimension**. For GPT-3 (d_model=12,288), the FFN inner dimension is 49,152.
- The activation function is **GeLU** (Gaussian Error Linear Unit) in most models, replacing the original paper's ReLU.
- **This is where most of the parameters live.** In a transformer, the FFN layers contain about 2/3 of all parameters.

**What does the FFN actually do?**

Think of attention as "gathering information from context" and the FFN as "processing that information." Research suggests that FFN layers act as **key-value memories** — they store factual knowledge learned during training. When you ask a model "What's the capital of France?", the answer "Paris" is likely retrieved from FFN weights.

### SwiGLU: The Modern FFN

Many modern models (LLaMA, Mistral, PaLM) use **SwiGLU**, a gated variant:

```
SwiGLU(x) = (x × W₁ × σ(x × W_gate)) × W₂
```

This adds a gate that controls information flow, improving performance. The tradeoff is a third weight matrix (W_gate), slightly increasing parameters.

---

### Residual Connections

Notice the "Add" steps in the block diagram. These are **residual connections** (also called skip connections):

```
output = LayerNorm(x) → Attention(x) + x
                                       ↑
                                 the original input is added back
```

**Why is this critical?**

Imagine stacking 96 transformer blocks (like GPT-3). Without residual connections, the gradients during training would have to flow through 96 transformations. They'd either vanish (become too small to learn from) or explode (become unstable).

Residual connections create a "highway" for gradients — they can flow directly from the output back to earlier layers through the skip connections:

```
  Block 1 ──→ Block 2 ──→ Block 3 ──→ ... ──→ Block 96 ──→ Output
     ↑           ↑           ↑                    ↑
     └───────────┴───────────┴────────────────────┘
           Gradient highway (residual connections)
```

This is what makes deep transformers possible to train.

---

## Stacking It All Together

A complete transformer stacks N identical blocks:

```
  Token Embeddings + Positional Encoding
                  │
                  ▼
          ┌──────────────┐
          │   Block 1    │
          └──────┬───────┘
                 │
          ┌──────────────┐
          │   Block 2    │
          └──────┬───────┘
                 │
                ...
                 │
          ┌──────────────┐
          │   Block N    │
          └──────┬───────┘
                 │
                 ▼
          Final Layer Norm
                 │
                 ▼
          Linear (to vocabulary)
                 │
                 ▼
          Softmax → Probability over tokens
```

The final linear layer projects from the model's hidden dimension (e.g., 768) to the vocabulary size (e.g., 50,257), producing a probability distribution over all possible next tokens.

### How Many Blocks?

| Model | Blocks (N) | d_model | Heads | Parameters |
|-------|-----------|---------|-------|------------|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-3 | 96 | 12,288 | 96 | 175B |
| LLaMA 2 7B | 32 | 4,096 | 32 | 7B |
| LLaMA 2 70B | 80 | 8,192 | 64 | 70B |

More blocks = deeper model = more capacity to learn complex patterns. But also more computation, more memory, and harder to train.

---

## How Text Generation Works

Let's trace through how a decoder-only transformer (like GPT) generates text:

**Prompt:** "The capital of France is"

```
Step 1: Tokenize
  → [464, 3139, 286, 4881, 318]

Step 2: Embed + add positions
  → 5 vectors of dimension d_model

Step 3: Pass through all N transformer blocks
  → Each block: self-attention → FFN (with residual + norm)
  → Output: 5 refined context-aware vectors

Step 4: Take the LAST token's output vector
  → Project to vocabulary size
  → Softmax → probability distribution

Step 5: Sample or pick the highest probability token
  → "Paris" (token ID 6342, probability 0.92)

Step 6: Append "Paris" to the sequence, repeat from Step 2
  → Now input is "The capital of France is Paris"
  → Next prediction: "." or "," etc.
```

This is **autoregressive generation** — one token at a time, each conditioned on all previous tokens.

---

## The KV-Cache: Making Generation Fast

There's an efficiency problem with autoregressive generation. Every time we generate a new token, we re-run the entire sequence through all blocks. But the attention computations for previous tokens haven't changed — only the new token is new.

The **KV-cache** stores the Key and Value matrices from previous tokens so they don't need to be recomputed:

```
Without KV-cache (wasteful):
  Step 1: Process ["The"]                           → 1 token
  Step 2: Process ["The", "capital"]                 → 2 tokens (re-processed "The")
  Step 3: Process ["The", "capital", "of"]           → 3 tokens (re-processed "The", "capital")
  ...

With KV-cache (efficient):
  Step 1: Process ["The"]                           → Cache K,V for "The"
  Step 2: Process ["capital"] + cached K,V          → Only 1 new token
  Step 3: Process ["of"] + cached K,V               → Only 1 new token
  ...
```

This is why inference gets **faster** after the initial "prefill" phase (processing the full prompt). Each subsequent token only requires computing attention for that one token against all cached keys and values.

But it's also why **long contexts use so much memory** — the KV-cache grows linearly with sequence length, and must be stored for every layer and every attention head.

```
KV-cache memory = 2 × num_layers × num_heads × head_dim × sequence_length × bytes_per_value
```

For LLaMA 2 70B with a 4K context, this is about 2.5 GB of KV-cache alone.

---

## A Complete Picture

Let's put everything together for a decoder-only transformer:

```
                    "The capital of France is ___"
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Token Embedding       │
                    │  + Positional Encoding │
                    └───────────┬───────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     ▼                      │
          │          ┌─────────────────────┐           │
          │          │     Layer Norm       │           │
          │          └──────────┬──────────┘           │
          │                     │                      │
          │          ┌──────────▼──────────┐           │
          │          │  Causal Multi-Head   │           │
          │          │  Self-Attention      │           │
          │          └──────────┬──────────┘           │
          │                     │                      │
   ×N     │              Add ←─┘ (residual)            │
  Blocks  │                     │                      │
          │          ┌──────────▼──────────┐           │
          │          │     Layer Norm       │           │
          │          └──────────┬──────────┘           │
          │                     │                      │
          │          ┌──────────▼──────────┐           │
          │          │  Feed-Forward (FFN)  │           │
          │          └──────────┬──────────┘           │
          │                     │                      │
          │              Add ←─┘ (residual)            │
          │                     │                      │
          └─────────────────────┼─────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Final Layer Norm    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Linear (→ vocab)     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Softmax          │
                    └───────────┬───────────┘
                                │
                                ▼
                    Probability: "Paris" = 92%
```

That's it. That's the entire architecture. Every modern LLM is a variation of this diagram.

---

## Key Takeaways

1. **A transformer block** = Layer Norm → Multi-Head Attention → Residual Add → Layer Norm → FFN → Residual Add
2. **Residual connections** are the highway that makes deep stacking possible
3. **FFN layers** store factual knowledge and contain most of the parameters
4. **Layer Norm** (pre-norm variant) stabilizes training
5. **KV-cache** makes autoregressive generation efficient but memory-hungry
6. **The architecture is surprisingly simple** — the power comes from scale and training data

---

## What's Next?

We've been focused on the decoder-only transformer. But the original paper had an encoder-decoder design, and BERT uses an encoder-only design. What are the differences? Why did decoder-only win for text generation?

Understanding the three flavors of transformers will help you choose the right tool for different tasks.

---

**Next: [Encoder vs Decoder — Three Flavors of Transformers](./05-encoder-vs-decoder.md)**

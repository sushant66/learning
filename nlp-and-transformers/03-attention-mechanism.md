# Chapter 3: The Attention Mechanism — The Idea That Changed Everything

## A Simple Observation

Read this sentence:

> "The animal didn't cross the street because **it** was too tired."

What does "it" refer to? The animal, obviously. You knew that instantly.

Now read this one:

> "The animal didn't cross the street because **it** was too wide."

Now "it" refers to the street. You switched your interpretation based on a single word at the end.

This is something humans do effortlessly: when processing a word, we **look at other words in the sentence** to understand its meaning. "It" by itself means nothing — its meaning comes entirely from context.

RNNs tried to do this through sequential processing, passing information forward step by step. But by the time you reach "tired" or "wide", the information about "animal" and "street" has been compressed and degraded through many steps.

What if, instead of passing information through a chain, every word could directly look at every other word?

That's self-attention.

---

## The Core Intuition

Imagine you're at a party (bear with me). You're trying to understand a conversation. At any moment, you might:
- Focus on the person currently speaking
- Glance at someone who was mentioned
- Ignore the background chatter

You're **selectively paying attention** to different sources of information based on what's relevant right now.

Self-attention works the same way. For each token in a sequence, the model asks:

> "Which other tokens in this sequence should I pay attention to in order to understand *this* token better?"

And it does this **for every token, simultaneously**.

---

## Query, Key, Value — The Mechanism

This is the part that seems intimidating at first but is actually straightforward once you see it.

Every token gets transformed into three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I carry?"

Think of it like a search engine:

```
You type a search query     →  Query
Each document has a title   →  Key
Each document has content   →  Value

The search engine compares your query against all titles (keys),
ranks them by relevance (attention scores),
and returns a weighted mix of their content (values).
```

### Step-by-Step Example

Let's trace through self-attention for a tiny sentence: **"The cat sat"**

Each token starts as an embedding vector (from the previous chapter). Let's say each is 4-dimensional for simplicity.

**Step 1: Create Q, K, V for each token**

Each token's embedding is multiplied by three learned weight matrices (W_Q, W_K, W_V) to produce its query, key, and value vectors.

```
         Embedding    ×  W_Q  =  Query       ×  W_K  =  Key         ×  W_V  =  Value
"The"  → [1,0,1,0]   →        → [0.2, 0.1]  →        → [0.1, 0.3]  →        → [1.0, 0.5]
"cat"  → [0,1,0,1]   →        → [0.8, 0.7]  →        → [0.9, 0.8]  →        → [0.3, 0.9]
"sat"  → [1,1,0,0]   →        → [0.5, 0.3]  →        → [0.4, 0.6]  →        → [0.7, 0.2]
```

(In reality, these dimensions are much larger — 64 to 128 per head.)

**Step 2: Compute attention scores**

For each token, compute how much it should attend to every other token by taking the **dot product** of its query with every key.

Let's compute attention scores for the word **"sat"**:

```
Score("sat" attending to "The") = Q_sat · K_The = [0.5,0.3] · [0.1,0.3] = 0.05 + 0.09 = 0.14
Score("sat" attending to "cat") = Q_sat · K_cat = [0.5,0.3] · [0.9,0.8] = 0.45 + 0.24 = 0.69
Score("sat" attending to "sat") = Q_sat · K_sat = [0.5,0.3] · [0.4,0.6] = 0.20 + 0.18 = 0.38
```

Higher score = more attention. "sat" attends most to "cat" (score 0.69).

**Step 3: Scale the scores**

Divide by √(key dimension) to keep gradients stable:

```
Scaled scores = [0.14, 0.69, 0.38] / √2 = [0.10, 0.49, 0.27]
```

**Step 4: Apply softmax**

Convert scores into a probability distribution (weights that sum to 1):

```
Softmax([0.10, 0.49, 0.27]) = [0.23, 0.42, 0.35]
                                 ↑      ↑      ↑
                               "The"  "cat"  "sat"
```

These are the **attention weights**. "sat" pays 42% attention to "cat", 35% to itself, and 23% to "The".

**Step 5: Weighted sum of values**

Multiply each value vector by its attention weight and sum them:

```
Output for "sat" = 0.23 × V_The  +  0.42 × V_cat  +  0.35 × V_sat
                 = 0.23 × [1.0, 0.5] + 0.42 × [0.3, 0.9] + 0.35 × [0.7, 0.2]
                 = [0.23, 0.12] + [0.13, 0.38] + [0.25, 0.07]
                 = [0.60, 0.56]
```

This output vector for "sat" is now **context-aware** — it contains information from all other tokens, weighted by relevance.

---

## The Formula

Everything above boils down to one elegant equation:

```
                        Q × K^T
Attention(Q, K, V) = softmax(─────────) × V
                         √d_k
```

Where:
- Q, K, V are matrices (all tokens' queries, keys, values stacked together)
- d_k is the dimension of the key vectors
- The whole thing runs as a single matrix multiplication — massively parallelizable on GPUs

```
  ┌──────────────────────────────────────────────────────┐
  │              Scaled Dot-Product Attention              │
  │                                                        │
  │   Q ──┐                                                │
  │        ├──→ MatMul ──→ Scale ──→ Softmax ──→ MatMul ──→ Output
  │   K ──┘     (Q×K^T)   (÷√d_k)   (weights)     ↑       │
  │                                                 │       │
  │   V ────────────────────────────────────────────┘       │
  │                                                        │
  └──────────────────────────────────────────────────────┘
```

---

## Why Scaling Matters

Why divide by √d_k? It seems like a small detail, but it's crucial.

When dimensions are large (say d_k = 64), dot products can produce very large numbers. Large numbers fed into softmax produce distributions that are almost one-hot — nearly all the attention goes to one token, and the gradients become tiny.

```
Without scaling (d_k = 64):
  Dot products might be: [45.2, 3.1, -12.7]
  Softmax:               [0.99, 0.00, 0.00]  ← almost all attention on one token
                                                 gradients vanish for the rest

With scaling (÷√64 = ÷8):
  Scaled:                [5.65, 0.39, -1.59]
  Softmax:               [0.84, 0.04, 0.01]  ← attention is more distributed
                                                 gradients flow to all tokens
```

---

## Multi-Head Attention: Multiple Perspectives

Here's the thing: a single attention computation learns **one type of relationship**. Maybe it learns that verbs attend to their subjects. But language has many types of relationships simultaneously:
- Syntactic (subject-verb agreement)
- Semantic (meaning relationships)
- Positional (nearby words)
- Referential (pronouns to their antecedents)

**Multi-head attention** runs multiple attention computations in parallel, each with its own Q, K, V weight matrices:

```
                    ┌──── Head 1: "Who did what?" (subject-verb)
                    │
  Input ────────────┼──── Head 2: "What describes what?" (adjective-noun)
  Embeddings        │
                    ├──── Head 3: "What refers to what?" (pronoun-antecedent)
                    │
                    └──── Head 4: "What's nearby?" (local context)

  Each head has its own W_Q, W_K, W_V matrices
  Each head operates on a smaller dimension (d_model / num_heads)
```

After all heads compute their outputs, the results are concatenated and projected:

```
  Head 1 output: [0.3, 0.1]
  Head 2 output: [0.7, 0.4]     → Concatenate → [0.3, 0.1, 0.7, 0.4, ...]
  Head 3 output: [0.2, 0.8]                              │
  Head 4 output: [0.5, 0.6]                              ▼
                                              Linear projection (W_O)
                                                          │
                                                          ▼
                                              Final output vector
```

Typical configurations:
- GPT-2: 12 heads, each with dimension 64 (total: 768)
- GPT-3: 96 heads, each with dimension 128 (total: 12,288)
- LLaMA 2 (70B): 64 heads, each with dimension 128 (total: 8,192)

**The key insight:** splitting into multiple heads doesn't increase computation much (each head works on a smaller dimension), but it lets the model learn different types of relationships simultaneously.

---

## Self-Attention vs Cross-Attention

There are actually two flavors of attention:

**Self-attention**: A sequence attends to itself. Every token looks at every other token in the same sequence. This is what we've been discussing.

```
  Input:  "The cat sat"

  "The" looks at → "The", "cat", "sat"
  "cat" looks at → "The", "cat", "sat"
  "sat" looks at → "The", "cat", "sat"
```

**Cross-attention**: One sequence attends to a different sequence. Used in encoder-decoder models for tasks like translation.

```
  Encoder (French): "Le chat est assis"
  Decoder (English): "The cat is"

  "is" looks at → "Le", "chat", "est", "assis"
  (The decoder attends to the encoder's output to decide what to generate next)
```

In decoder-only models (GPT, Claude, LLaMA), only self-attention is used. This keeps things simpler and, as it turns out, works remarkably well.

---

## Causal (Masked) Attention

There's one more critical detail. When generating text, the model predicts one token at a time: "The" → "cat" → "sat" → ...

When predicting "sat", should it be allowed to see "sat" and everything after it? No — that would be cheating. It should only see what came before.

**Causal attention** (also called masked attention) prevents tokens from attending to future tokens:

```
                    Attending to:
                   "The"  "cat"  "sat"  "on"
  "The" can see:  [  ✓      ✗      ✗     ✗  ]
  "cat" can see:  [  ✓      ✓      ✗     ✗  ]
  "sat" can see:  [  ✓      ✓      ✓     ✗  ]
  "on"  can see:  [  ✓      ✓      ✓     ✓  ]
```

This is implemented by setting the "future" attention scores to negative infinity before the softmax, which makes them effectively zero:

```
  Raw scores:     [0.3,  0.7,  0.5,  0.2]
  After masking:  [0.3,  0.7,  -∞,   -∞ ]
  After softmax:  [0.40, 0.60, 0.00, 0.00]
```

This is why these models are called **autoregressive** — each prediction depends only on previous tokens.

---

## Why Self-Attention Won

Let's compare self-attention to the alternatives we saw in Chapter 1:

| Property | RNN/LSTM | Self-Attention |
|----------|----------|----------------|
| Sees all tokens | Through a chain (indirect) | Directly (one step) |
| Parallelizable | No (must go step by step) | Yes (all at once) |
| Long-range dependencies | Degrades over distance | Equal access to all positions |
| Computation per layer | O(n) | O(n²) per layer |

The O(n²) cost is self-attention's weakness — every token attends to every other token, so doubling the sequence length quadruples the computation. This is why context windows are finite and why there's active research on more efficient attention variants.

But for sequences of moderate length, the benefits overwhelmingly outweigh the cost. Self-attention is:
- **More parallelizable** → trains faster on GPUs
- **Better at long-range dependencies** → understands context better
- **Simpler** → fewer architectural tricks needed

---

## What You Just Learned

The attention mechanism is the core innovation of the transformer. Everything else — the feed-forward layers, the normalization, the residual connections — is supporting infrastructure. If you understand attention, you understand the heart of every modern LLM.

To recap:
1. Each token creates a **Query** ("what am I looking for?"), **Key** ("what do I offer?"), and **Value** ("what information do I carry?")
2. Attention scores = dot product of queries and keys, scaled and softmaxed
3. Output = weighted sum of values
4. **Multi-head attention** lets the model learn multiple types of relationships
5. **Causal masking** prevents looking at future tokens during generation

---

## What's Next?

Attention is the star of the show, but a transformer is more than just attention. There are feed-forward networks, layer normalization, residual connections, and a specific way all these pieces are stacked together.

In the next chapter, we'll zoom out and see how the complete transformer architecture is assembled — block by block, layer by layer.

---

**Next: [The Transformer Architecture — Putting It All Together](./04-transformer-architecture.md)**

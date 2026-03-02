# Chapter 2: Tokenization — How Text Becomes Numbers

## The Question That Seems Too Simple

You type "Hello, world!" into ChatGPT. Within seconds, you get a response. But here's a question most people never ask:

**How does the model actually read your text?**

It doesn't see letters. It doesn't see words. Neural networks only understand numbers — specifically, vectors of floating-point numbers. So somewhere between you typing "Hello, world!" and the model processing it, your text has to be converted into numbers.

That conversion process is called **tokenization**, and it's far more interesting than it sounds. The choices made during tokenization directly affect what the model can learn, how fast it runs, and how much it costs you to use.

---

## Why Not Just Use Words?

The most intuitive approach: split text by spaces and give each word an ID.

```
"The cat sat on the mat" → ["The", "cat", "sat", "on", "the", "mat"]
                         → [  42,   891,  1204,  15,    42,   3302 ]
```

Simple. But it breaks down fast.

**Problem 1: The vocabulary explodes.**

English has roughly 170,000 words in active use. Add names, technical terms, slang, misspellings, and other languages — you're looking at millions of possible words. Each word needs its own row in an embedding table (a giant matrix), and that table needs to fit in GPU memory.

**Problem 2: Unknown words.**

What happens when the model encounters "ChatGPT" for the first time? Or "defenestration"? Or a typo like "teh"? With word-level tokenization, the answer is: it can't handle them. You'd have to use a special `[UNK]` (unknown) token and lose all information about that word.

**Problem 3: Morphology is ignored.**

"run", "running", "runner", and "runs" are four separate entries with no shared knowledge. The model has to independently learn what each one means, even though they clearly share a root.

---

## Why Not Just Use Characters?

Go to the other extreme: split everything into individual characters.

```
"cat" → ["c", "a", "t"] → [3, 1, 20]
```

Tiny vocabulary (just ~256 characters for English + symbols). No unknown tokens ever. But...

**Problem: Sequences become extremely long.**

The sentence "The cat sat on the mat" is 6 tokens with words but **22 tokens** with characters (including spaces). Longer sequences mean:
- More computation (attention is quadratic in sequence length)
- Harder to learn long-range patterns
- Much higher cost

The model has to learn that "c", "a", "t" together mean a furry animal. That's asking it to do a lot of extra work.

---

## The Goldilocks Zone: Subword Tokenization

What if we split text into pieces that are somewhere between words and characters?

Common words stay whole. Rare words get broken into smaller, reusable pieces.

```
"unhappiness" → ["un", "happiness"]
"ChatGPT"     → ["Chat", "G", "PT"]
"running"     → ["run", "ning"]
"the"         → ["the"]
```

This is **subword tokenization**, and it's what every modern LLM uses. It gives you:
- A manageable vocabulary size (32K to 100K tokens)
- No unknown tokens (any word can be broken into known pieces)
- Shared knowledge between related words ("run" appears in "running", "runner", etc.)

The question is: how do you decide where to split?

---

## Byte Pair Encoding (BPE)

BPE is the most popular subword tokenization algorithm. GPT-2, GPT-3, GPT-4, and LLaMA all use variants of it.

The idea is beautifully simple. Start with individual characters and repeatedly merge the most frequent pair.

### How BPE Works — Step by Step

Let's say our training text contains these words (with frequencies):

```
"low"    → 5 times
"lower"  → 2 times
"newest" → 6 times
"widest" → 3 times
```

**Step 0:** Start with characters as our initial tokens.

```
Vocabulary: { l, o, w, e, r, n, s, t, i, d }
```

**Step 1:** Find the most frequent adjacent pair across all words.

```
l o w           → "l o" appears 7 times (5+2), "o w" appears 7 times
n e w e s t     → "e s" appears 9 times (6+3) ← most frequent!
w i d e s t     → "s t" appears 9 times (6+3) ← tie!
```

Let's pick "e s" → merge into "es"

```
Vocabulary: { l, o, w, e, r, n, s, t, i, d, es }
```

**Step 2:** Repeat. Find the next most frequent pair.

Now "es t" appears 9 times → merge into "est"

```
Vocabulary: { l, o, w, e, r, n, s, t, i, d, es, est }
```

**Step 3:** Continue. "l o" appears 7 times → merge into "lo"

```
Vocabulary: { l, o, w, e, r, n, s, t, i, d, es, est, lo }
```

Keep going until you reach your desired vocabulary size (typically 32K-100K merges).

The result: common words and subwords get their own tokens, while rare words are composed of smaller pieces.

```
After training:
"lowest"    → ["low", "est"]
"newest"    → ["new", "est"]
"widest"    → ["wid", "est"]
"the"       → ["the"]              ← common word, single token
"indubitably" → ["in", "dub", "it", "ably"]  ← rare word, multiple tokens
```

### Why BPE is Elegant

Notice how "est" became its own token. The model can now understand the concept of superlatives (-est) as a reusable building block. It doesn't need to independently learn what "newest", "widest", "tallest", "fastest" each mean — they all share the "est" token.

---

## SentencePiece

BPE assumes you've already split text into words (usually by spaces). But what about languages like Chinese or Japanese that don't use spaces? Or what about treating the raw text as-is?

**SentencePiece** solves this by treating the input as a raw stream of characters — including spaces. It uses a special character `▁` (Unicode lower bar) to mark word boundaries:

```
"I like cats" → ["▁I", "▁like", "▁cats"]
"東京は" → ["▁東京", "は"]
```

SentencePiece can run BPE or another algorithm called **Unigram** on this raw stream. It's used by T5, LLaMA, and many multilingual models because it's language-agnostic.

---

## tiktoken: OpenAI's Tokenizer

OpenAI uses a BPE variant implemented in a library called **tiktoken**. It's blazing fast (written in Rust) and used by all GPT models.

Different models use different tokenizers with different vocabulary sizes:

```
GPT-2:       50,257 tokens
GPT-3.5/4:  100,277 tokens (cl100k_base)
GPT-4o:     200,019 tokens (o200k_base)
```

You can try it yourself:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

text = "Tokenization is fascinating!"
tokens = enc.encode(text)

print(tokens)        # [3947, 2065, 374, 27443, 0]
print(len(tokens))   # 5 tokens

# See what each token looks like
for t in tokens:
    print(f"  {t} → '{enc.decode([t])}'")

# Output:
#   3947 → 'Token'
#   2065 → 'ization'
#   374  → ' is'
#   27443 → ' fascinating'
#   0    → '!'
```

Notice: "Tokenization" was split into "Token" + "ization". The space before "is" is part of the token " is". This is how GPT models see your text.

---

## Special Tokens

Every tokenizer includes special tokens that aren't regular words but carry important signals:

```
[CLS]           → "This is the start of a classification input" (BERT)
[SEP]           → "This separates two segments" (BERT)
<|endoftext|>   → "This is where the text ends" (GPT)
<s> / </s>      → "Start/end of sequence" (LLaMA, T5)
<|im_start|>    → "Start of a chat message" (ChatGPT)
[PAD]           → "Ignore this, it's just padding to fill the batch"
```

These tokens are added by the tokenizer automatically. When you send a message to ChatGPT, behind the scenes it might look like:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is tokenization?<|im_end|>
<|im_start|>assistant
```

---

## Why Vocabulary Size Matters

This isn't just a technical detail — it directly affects cost and performance.

### Larger Vocabulary (100K+ tokens)

```
Pros:
  + Common words are single tokens → shorter sequences
  + Faster inference (fewer tokens to generate)
  + Cheaper (APIs charge per token)

Cons:
  - Bigger embedding table (more GPU memory)
  - Rare tokens might not get enough training data
```

### Smaller Vocabulary (32K tokens)

```
Pros:
  + Smaller model size
  + Every token is well-trained (seen enough times)

Cons:
  - More tokens needed per sentence → slower, more expensive
  - Less efficient for common phrases
```

### A Real-World Example

```
Text: "The transformer architecture is revolutionary."

With a 32K vocabulary (more splits):
  → ["The", " transform", "er", " architecture", " is", " revolution", "ary", "."]
  → 8 tokens

With a 100K vocabulary (fewer splits):
  → ["The", " transformer", " architecture", " is", " revolutionary", "."]
  → 6 tokens
```

At scale, this difference is enormous. If you're processing millions of requests, 25% fewer tokens means 25% lower cost.

---

## The Token ≠ Word Trap

One of the most common mistakes when working with LLMs:

**Tokens are not words.** They're not characters either. They're an in-between.

A rough rule of thumb for English: **1 token ≈ 0.75 words** (or about 4 characters).

This matters because:
- **Context windows** are measured in tokens, not words. A 128K context window is roughly 96K words.
- **API pricing** is per token. More verbose prompts cost more.
- **Token limits** can cut off your text mid-word if you hit the maximum.

```
"I" = 1 token
"I'm" = 1 token (common enough to be its own token)
"counterrevolutionary" = 3 tokens ("counter", "revolution", "ary")
"🎉" = 1-2 tokens (depends on the tokenizer)
"こんにちは" = 1-3 tokens (depends on multilingual support)
```

---

## The Full Pipeline: Text to Numbers

Here's what happens when you type a prompt into an LLM:

```
   "The cat sat on the mat"
              │
              ▼
   ┌─────────────────────┐
   │     Tokenizer        │  Split text into subword tokens
   └──────────┬──────────┘
              │
              ▼
   ["The", " cat", " sat", " on", " the", " mat"]
              │
              ▼
   ┌─────────────────────┐
   │    Token → ID        │  Look up each token in vocabulary
   └──────────┬──────────┘
              │
              ▼
   [464, 3797, 3290, 319, 262, 2603]
              │
              ▼
   ┌─────────────────────┐
   │  Embedding Lookup    │  Convert each ID to a dense vector
   └──────────┬──────────┘
              │
              ▼
   [ [0.12, -0.34, 0.56, ...],    ← 768 or more dimensions
     [0.78, 0.23, -0.11, ...],      each per token
     [0.45, -0.67, 0.89, ...],
     ... ]
              │
              ▼
   ┌─────────────────────┐
   │  Transformer Model   │  Process these vectors
   └─────────────────────┘
```

The embedding lookup table is a matrix of shape `(vocab_size, embedding_dim)`. For GPT-3, that's `50,257 × 12,288` — about 617 million numbers just for the embeddings.

---

## Key Takeaways

1. **Subword tokenization** (BPE, SentencePiece) is the standard. It balances vocabulary size with sequence length.
2. **Tokens ≠ words.** This affects context windows, pricing, and how you think about model inputs.
3. **The tokenizer is trained separately** from the model, on its own corpus, before model training begins.
4. **Vocabulary size is a design choice** that trades off memory, speed, and representation quality.
5. **Special tokens** mark boundaries and structure that the model needs to understand.

---

## What's Next?

We now know how text becomes a sequence of token vectors. But here's the question that kept NLP researchers up at night for years:

How do you make each token aware of *all the other tokens* around it — not just the ones immediately next to it, but tokens that might be hundreds of positions away?

The answer is a mechanism so powerful it replaced everything that came before it. It's called **self-attention**, and it's the beating heart of the transformer.

---

**Next: [The Attention Mechanism — The Idea That Changed Everything](./03-attention-mechanism.md)**

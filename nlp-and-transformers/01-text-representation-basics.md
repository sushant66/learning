# Chapter 1: Text Representation — Before Transformers

## The Fundamental Problem

Here's something we take for granted: you're reading this sentence and you *understand* it. You know what each word means, how they relate to each other, and what the sentence is trying to say.

A computer sees this:

```
01010100 01101000 01100101 00100000 01100011 01100001 01110100 ...
```

Just bytes. No meaning. No context. No understanding.

So the first challenge in all of NLP (Natural Language Processing) is this: **how do you represent words as numbers in a way that preserves meaning?**

Every approach in the history of NLP is an attempt to answer that question. Let's walk through them — from the simplest to the one that set the stage for transformers.

---

## Bag of Words: Counting What's There

The simplest idea: just count how many times each word appears.

Take two sentences:

```
Sentence A: "The cat sat on the mat"
Sentence B: "The dog sat on the log"
```

Build a vocabulary from all unique words, then count:

```
         the  cat  sat  on  mat  dog  log
Sent A: [ 2,   1,   1,  1,   1,   0,   0 ]
Sent B: [ 1,   0,   1,  1,   0,   1,   1 ]
```

Each sentence becomes a vector of numbers. You can now compare them, classify them, do math with them.

**What's good about this?**
- Simple. Fast. Easy to implement.

**What's wrong with this?**
- **Word order is completely lost.** "Dog bites man" and "Man bites dog" get the same representation.
- **No sense of meaning.** "Happy" and "joyful" are treated as completely unrelated.
- **Vectors are huge and sparse.** A vocabulary of 100,000 words means 100,000-dimensional vectors, mostly filled with zeros.

### TF-IDF: A Small Improvement

TF-IDF (Term Frequency - Inverse Document Frequency) improves on bag of words by asking: **"Is this word actually important in this document, or does it just appear everywhere?"**

- **TF**: How often does this word appear in *this* document?
- **IDF**: How rare is this word across *all* documents?

Words like "the" and "is" appear everywhere, so they get low scores. Words like "transformer" or "gradient" appear in fewer documents, so they get high scores.

```
TF-IDF("the")         = High TF  ×  Low IDF  = Low score  (common, unimportant)
TF-IDF("transformer") = Some TF  ×  High IDF = High score (rare, informative)
```

It's better, but we still have the same fundamental problems: no word order, no sense of meaning.

---

## Word2Vec: Words as Vectors with Meaning

In 2013, Tomas Mikolov at Google introduced an idea that felt almost magical: **what if words that appear in similar contexts have similar meanings?**

Think about it. You've never been told the definition of every word you know. You learned most words by seeing them used in context, over and over. Word2Vec does the same thing.

It trains a small neural network on a massive amount of text with a simple task:

> Given a word, predict the words around it (or vice versa).

After training, each word gets a dense vector — typically 100 to 300 dimensions. And these vectors capture meaning.

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

That's not a trick. It actually works. The model learned that "king" is to "man" as "queen" is to "woman" — just from reading text.

```
        Meaning Space (simplified to 2D)

    queen •
                    • woman
    king •
                    • man

         ← royalty →
```

**What's good about this?**
- Words with similar meanings have similar vectors.
- Vectors are dense and compact (300 dimensions vs 100,000).
- You can do arithmetic with meaning.

**What's wrong with this?**
- **Each word gets ONE vector, regardless of context.** The word "bank" has the same vector whether you're talking about a river bank or a financial bank.
- **It's still word-level.** You get a vector for each word, but how do you represent a whole sentence?

### GloVe: A Cousin of Word2Vec

GloVe (Global Vectors for Word Representation) is another way to create word embeddings. Instead of a neural network, it uses a co-occurrence matrix — a giant table counting how often words appear near each other.

The result is similar: dense vectors that capture meaning. Word2Vec and GloVe are often mentioned together because they solve the same problem in slightly different ways.

---

## RNNs & LSTMs: Reading One Word at a Time

The approaches above treat words in isolation. But language is sequential — the meaning of a word depends on what came before it.

**Recurrent Neural Networks (RNNs)** process text one word at a time, maintaining a "hidden state" that acts like a running memory.

```
  "The"  →  "cat"  →  "sat"  →  "on"  →  "the"  →  "mat"
    ↓         ↓         ↓         ↓         ↓         ↓
  ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐
  │ h₁│───→│ h₂│───→│ h₃│───→│ h₄│───→│ h₅│───→│ h₆│───→ output
  └───┘    └───┘    └───┘    └───┘    └───┘    └───┘
         hidden state carries forward
```

Each step takes the current word *and* the previous hidden state, producing a new hidden state. In theory, the final hidden state encodes the meaning of the entire sentence.

**The Problem: Vanishing Gradients**

In practice, RNNs have a fatal flaw. When sentences get long, the hidden state has to carry information across many steps. During training (backpropagation through time), gradients either vanish (become nearly zero) or explode (become enormous).

This means **RNNs effectively forget what happened at the beginning of a long sentence** by the time they reach the end.

### LSTMs: A Better Memory

**Long Short-Term Memory (LSTM)** networks were designed to fix this. They add "gates" — mechanisms that control what information to keep, what to forget, and what to output.

```
  ┌─────────────────────────────────────────┐
  │              LSTM Cell                   │
  │                                          │
  │  ┌──────┐  ┌──────┐  ┌──────┐           │
  │  │Forget│  │Input │  │Output│           │
  │  │ Gate │  │ Gate │  │ Gate │           │
  │  └──┬───┘  └──┬───┘  └──┬───┘           │
  │     ↓         ↓         ↓               │
  │  ════════ Cell State (long-term) ═══════│
  │                                          │
  │  ──────── Hidden State (short-term) ────│
  └─────────────────────────────────────────┘
```

- **Forget Gate**: What old information should we throw away?
- **Input Gate**: What new information should we store?
- **Output Gate**: What should we output right now?

LSTMs were the state of the art for years. Machine translation, text generation, sentiment analysis — all powered by LSTMs.

**But they had two major problems:**

1. **Sequential processing.** You *must* process word 1 before word 2 before word 3. You can't parallelize. Training is slow.
2. **Long-range dependencies are still hard.** Even with gates, information degrades over very long sequences. By the time you reach word 500, the context from word 1 is faint.

---

## Seq2Seq with Attention: The Bridge

For tasks like translation, researchers used an **encoder-decoder** setup:

```
  Encoder (reads input)          Decoder (generates output)

  "The"→"cat"→"sat"  ──→  [context vector]  ──→  "Le"→"chat"→"assis"
```

The encoder reads the input sentence and compresses it into a single context vector. The decoder then generates the output from that vector.

**The bottleneck:** Everything the encoder understood has to fit into *one* fixed-size vector. For long sentences, this is like trying to summarize a book into a single tweet.

### Attention: Let the Decoder Look Back

In 2014, Bahdanau et al. introduced **attention**: instead of relying on a single context vector, let the decoder look at *all* encoder hidden states and decide which ones are most relevant at each step.

```
  Encoder states:    h₁    h₂    h₃    h₄    h₅
                      ↑     ↑     ↑↑    ↑     ↑
                      │     │     ││    │     │
  Attention weights: 0.05  0.1   0.6   0.15  0.1
                                  ↑
                          "This word matters most
                           for the current output"
```

At each decoding step, the model assigns a weight to every encoder state. High weight = "pay attention to this word." Low weight = "ignore this."

This was a game-changer. Translation quality jumped. The model could now handle long sentences because it could always look back at any part of the input.

**But here's the key insight that led to transformers:** if attention works so well *between* encoder and decoder, what if we used attention *within* a single sequence? What if every word could attend to every other word?

That idea — **self-attention** — is what the transformer is built on.

---

## The Timeline: How We Got Here

```
  1950s-80s    Rule-based NLP (hand-written grammar rules)
      ↓
  1990s-2000s  Statistical NLP (bag of words, TF-IDF, n-grams)
      ↓
  2013         Word2Vec / GloVe (meaningful word vectors)
      ↓
  2014-2016    RNNs / LSTMs + Attention (sequence modeling)
      ↓
  2017         Transformers ("Attention Is All You Need")
      ↓
  Everything changed.
```

Each step solved a problem the previous approach couldn't handle. Transformers didn't come out of nowhere — they're the culmination of decades of work.

---

## Key Takeaways

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| Bag of Words / TF-IDF | Simple, fast | No word order, no meaning |
| Word2Vec / GloVe | Captures meaning | One vector per word (no context) |
| RNNs / LSTMs | Handles sequences | Slow, forgets long-range info |
| Seq2Seq + Attention | Looks at all inputs | Still sequential, encoder bottleneck |

Every weakness in this table is something the transformer architecture solves. That's why it won.

---

## What's Next?

Before we can understand transformers, we need to answer a more basic question: when a model reads "The cat sat on the mat," it doesn't see words — it sees **tokens**. What are tokens? How does text get split up? And why does it matter?

That's what tokenization is all about.

---

**Next: [Tokenization — How Text Becomes Numbers](./02-tokenization.md)**

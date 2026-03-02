# Chapter 6: Pre-training Objectives — How Models Learn Language

## The Bootstrap Problem

Here's a paradox: we want to build a model that understands language. To train it, we'd normally need labeled data — millions of examples with correct answers. But creating that much labeled data by hand is impossible.

So how did GPT-3 learn from 300 billion tokens of text without a single human label?

The answer is one of the most elegant ideas in AI: **self-supervised learning**. You use the text itself to create the training signal. No labels needed.

The way you set up this "learn from raw text" task is called the **pre-training objective**, and it turns out the choice of objective fundamentally shapes what the model becomes good at.

---

## Causal Language Modeling (CLM) — "Predict the Next Word"

This is the objective behind GPT, Claude, LLaMA, and every decoder-only model.

**The idea is absurdly simple:** given a sequence of words, predict the next one.

```
Input:    "The cat sat on the"
Target:   "mat"

Input:    "Machine learning is a subset of"
Target:   "artificial"

Input:    "def fibonacci(n):"
Target:   "\n"
```

During training, the model sees trillions of these examples, extracted from books, websites, code, and conversations. For every position in every text, it tries to predict what comes next.

### Why This Works So Incredibly Well

Think about what you need to know to predict the next word in these sentences:

```
"The Eiffel Tower is located in ___"
→ You need to know geography.

"She felt happy because her experiment ___"
→ You need to understand causality and emotions.

"In Python, to open a file you use the ___ function"
→ You need to know programming.

"The patient presented with fever, cough, and ___"
→ You need medical knowledge.

"2 + 2 = ___"
→ You need arithmetic.
```

Next-token prediction forces the model to learn an enormous amount about the world — facts, reasoning, grammar, style, code, math — just to get better at predicting.

It's like studying for an exam by trying to guess the next sentence in every textbook. You'd end up learning the actual material.

### The Training Process

```
Training text: "The cat sat on the mat"

Position 1: Input ["The"]                  → Predict "cat"    ✓ or ✗
Position 2: Input ["The", "cat"]           → Predict "sat"    ✓ or ✗
Position 3: Input ["The", "cat", "sat"]    → Predict "on"     ✓ or ✗
Position 4: Input ["The", "cat", "sat", "on"] → Predict "the" ✓ or ✗
Position 5: Input ["The", "cat", "sat", "on", "the"] → Predict "mat" ✓ or ✗
```

Thanks to causal masking, all five predictions happen **simultaneously** in a single forward pass. The model processes the entire sequence at once, but each position can only see tokens before it.

The loss function is **cross-entropy**: it measures how far the model's predicted probability distribution is from the actual next token. If the model assigns 90% probability to "mat" and the true token is "mat", the loss is low. If it assigns only 2%, the loss is high.

```
Model prediction: P("mat") = 0.02, P("dog") = 0.15, P("car") = 0.10, ...
Actual answer:    "mat"
Loss:             -log(0.02) = 3.91  ← high loss, model was wrong

After training:
Model prediction: P("mat") = 0.85, P("dog") = 0.03, P("car") = 0.01, ...
Loss:             -log(0.85) = 0.16  ← low loss, model learned
```

### What CLM Can't Do Well

Because the model only sees left-to-right context, it never learns to use future context during pre-training. The word "it" in "The **animal** didn't cross the street because **it** was too tired" can only attend to tokens before it — it can see "animal" but not "tired."

For pure understanding tasks (classification, extracting information), this one-directional view is a handicap compared to bidirectional models. But it's the price you pay for being able to generate text.

---

## Masked Language Modeling (MLM) — "Fill in the Blanks"

This is the objective behind BERT, RoBERTa, and other encoder-only models.

**The idea:** randomly hide some tokens and ask the model to predict them.

```
Original:  "The cat sat on the mat"
Masked:    "The [MASK] sat on the [MASK]"
Predict:   [MASK₁] = "cat", [MASK₂] = "mat"
```

### How Masking Works

BERT masks 15% of input tokens, but not all of them are replaced with [MASK]:
- 80% are replaced with [MASK]
- 10% are replaced with a random token
- 10% are left unchanged

```
Original: "I love machine learning"

80% case: "I love [MASK] learning"      → predict "machine"
10% case: "I love banana learning"       → predict "machine" (random replacement)
10% case: "I love machine learning"      → predict "machine" (unchanged)
```

**Why the 80/10/10 split?**

If you always used [MASK], the model would learn that [MASK] = "something to predict" and would never see [MASK] during actual use (inference). The random replacement and unchanged cases force the model to maintain good representations for all tokens, not just masked ones.

### The Advantage: Bidirectional Context

To predict [MASK] in "The [MASK] sat on the mat", the model can look at:
- "The" (before) — suggests an article + noun
- "sat on the mat" (after) — suggests an animal or person

Both directions contribute. This is why BERT-style models are better than GPT-style for pure understanding tasks like classification and entity extraction.

### The Limitation

MLM models can't generate text naturally. They're trained to fill in blanks, not to produce sequences from left to right. You can't easily ask BERT to "continue this story."

---

## Span Corruption — "Fix the Missing Pieces"

This is the objective behind T5 (Text-to-Text Transfer Transformer).

Instead of masking individual tokens, T5 masks **contiguous spans** and replaces them with sentinel tokens:

```
Original:  "The cat sat on the mat and then went home"
Corrupted: "The <X> on the mat <Y> went home"
Target:    "<X> cat sat <Y> and then"
```

The model must generate the missing spans as a sequence.

### Why Spans Instead of Single Tokens?

- Single-token masking (BERT) means most predictions are easy — there are limited options for one word.
- Span corruption forces the model to generate **multi-token sequences**, which is closer to actual text generation.
- It's also more efficient — fewer sentinel tokens means shorter sequences.

### The Text-to-Text Framing

T5's big insight was framing **every** NLP task as text-to-text:

```
Translation:
  Input:  "translate English to German: That is good"
  Output: "Das ist gut"

Summarization:
  Input:  "summarize: [long article text]"
  Output: "Article summary here"

Classification:
  Input:  "sentiment: This movie was amazing"
  Output: "positive"

Question Answering:
  Input:  "question: What color is the sky? context: The sky is blue."
  Output: "blue"
```

One format, one model, any task. This was a precursor to the "prompt everything" approach that decoder-only models later adopted.

---

## Comparing the Three Objectives

```
  Causal LM (GPT)              Masked LM (BERT)           Span Corruption (T5)

  "The cat sat ___"            "The [M] sat on [M]"       "The <X> on the mat"
       ↓                            ↓                           ↓
  Predict next token           Predict masked tokens       Generate missing spans
       ↓                            ↓                           ↓
  Left-to-right only           Bidirectional               Both (enc-dec)
       ↓                            ↓                           ↓
  Can generate text            Can't generate easily       Can generate text
       ↓                            ↓                           ↓
  GPT, Claude, LLaMA           BERT, RoBERTa              T5, BART
```

| Property | CLM | MLM | Span Corruption |
|----------|-----|-----|-----------------|
| Context direction | Left only | Both | Both (encoder) |
| Can generate text | Yes | No | Yes |
| Training efficiency | Every token is a prediction | Only 15% of tokens | Only corrupted spans |
| Dominant models | GPT, Claude, LLaMA | BERT, RoBERTa | T5, FLAN-T5 |

---

## From Pre-training to Usefulness

Pre-training gives the model general language understanding. But a pre-trained model is like a well-read person who hasn't been told what job they're doing. It can predict the next token brilliantly, but it doesn't know how to follow instructions or have a conversation.

That's where **fine-tuning** comes in, and it happens in stages:

```
  Stage 1: Pre-training (Self-supervised)
  ────────────────────────────────────────
  Data:      Trillions of tokens from the internet
  Objective: Predict next token (CLM)
  Result:    A model that "understands" language

         ↓

  Stage 2: Supervised Fine-Tuning (SFT)
  ────────────────────────────────────────
  Data:      Thousands of (instruction, response) pairs
             Written by humans
  Objective: Learn to follow instructions
  Result:    A model that can answer questions, write code, etc.

         ↓

  Stage 3: RLHF / DPO (Alignment)
  ────────────────────────────────────────
  Data:      Human preferences ("Response A is better than B")
  Objective: Align with human values and preferences
  Result:    A model that's helpful, harmless, and honest

         ↓

  The model you actually interact with (ChatGPT, Claude, etc.)
```

We'll cover fine-tuning in detail in later phases. For now, the key point is: pre-training is the foundation. It takes months and millions of dollars. Everything else builds on top of it.

---

## The Numbers Are Staggering

To give you a sense of scale:

```
GPT-3 pre-training:
  - 300 billion tokens of training data
  - 175 billion parameters
  - Trained on 10,000 GPUs
  - Estimated cost: $4.6 million
  - Training time: ~34 days

LLaMA 2 70B pre-training:
  - 2 trillion tokens of training data
  - 70 billion parameters
  - Trained on 2,048 A100 GPUs
  - Estimated cost: $2-3 million
  - Training time: ~21 days

GPT-4 pre-training (estimated):
  - 13+ trillion tokens
  - 1.7 trillion parameters (rumored)
  - Cost: $50-100 million
```

This is why pre-training is done by a handful of companies with massive compute budgets. Most GenAI engineers will work with pre-trained models, fine-tuning them or building applications on top.

---

## Key Takeaways

1. **Pre-training is self-supervised** — the model creates its own training signal from raw text.
2. **CLM (next-token prediction)** is the dominant objective because it enables generation and scales well.
3. **MLM (masked prediction)** gives better understanding but can't generate text.
4. **Span corruption** combines both capabilities in an encoder-decoder setup.
5. **Pre-training is just the first step** — SFT and RLHF turn a raw model into a useful assistant.
6. **Pre-training is enormously expensive** — this is why most engineers work with existing pre-trained models.

---

## What's Next?

Now that you understand how different models are trained, let's meet the models themselves. BERT, GPT, T5, LLaMA — each one made specific architectural and training choices that shaped the field. Understanding their differences will help you pick the right model for any task.

---

**Next: [Key Model Families — The Models That Shaped the Field](./07-key-model-families.md)**

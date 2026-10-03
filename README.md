# ML Interview Math

**First-principles ML/LLM interview questions, derivations, systems notes, and follow-ups.**

This repository is the public companion to the daily [@MLInterviewMath](https://x.com/MLInterviewMath) series.

I've passed ML interview loops at FAANG and frontier labs, and now run interviews as a hiring manager. In my experience, the fundamentals round increasingly expects candidates to derive, quantify, and reason from first principles rather than recite definitions.

## Start here

| # | Question | Topic | Status |
|---|---|---|---|
| 001 | [Why does attention divide by √dₖ?](questions/001-attention-scaling.md) | Attention | Published |
| 002 | When does the N² attention term dominate? | MHA / FLOPs | Next |
| 003 | Why is causal training parallel? | Causal attention | Planned |
| 004 | How does RMSNorm change the backward geometry? | Normalization | Planned |
| 005 | Why does SwiGLU use a width near 8d/3? | FFN | Planned |
| 006 | Why does RoPE produce relative-position-dependent scores? | Position encoding | Planned |
| 007 | How large is the KV cache? | Serving | Planned |
| 008 | How can FlashAttention compute exact softmax block-wise? | GPU / IO | Planned |
| 009 | Where does MoE move the bottleneck? | MoE / distributed systems | Planned |

See the [Transformer question map](topics/transformers.md).

## What these questions are designed to test

The goal is not to memorize model trivia. Each question follows a repeatable interview workflow:

1. **Write the object precisely.** Shapes, assumptions, and objective.
2. **Derive the core result.** Reconstruct it rather than quote it.
3. **Quantify the scaling.** Parameters, FLOPs, memory, variance, or communication.
4. **Connect the math to implementation.** What changes on real hardware?
5. **Handle the follow-up.** Which assumption breaks? What trade-off changes?

## Repository structure

    questions/   Worked interview questions and follow-ups
    code/        Small reproducible experiments and calculators
    topics/      Topic-based navigation
    book/        About the book and errata
    .github/     Issue templates for corrections and question suggestions

## Companion book: *Underneath the Surface*

**First-Principles ML & LLM Interview Prep for Frontier Labs**

The repo is deliberately question-by-question. The book is the structured reference: mathematical foundations, classical ML, autograd, Transformer mechanics, pretraining, post-training, distributed systems, and an interview toolkit.

[What is in the book →](book/about.md)

## Run the Day 1 experiment

    python -m pip install -r requirements.txt
    python code/001_attention_scaling.py

The experiment measures how attention entropy changes with head dimension under three choices: no scaling, division by √dₖ, and division by dₖ.

## Corrections and suggestions

Technical corrections are welcome. Please use the [correction template](.github/ISSUE_TEMPLATE/correction.md).

Have a good interview question or follow-up? Use the [question suggestion template](.github/ISSUE_TEMPLATE/question-suggestion.md).

Please do not submit confidential, proprietary, leaked, or NDA-covered interview content.

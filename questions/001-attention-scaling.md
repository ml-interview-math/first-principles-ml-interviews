# ML Interview Question #001 — Why does attention divide by √dₖ?

> **Topic:** Transformer mechanics / scaled dot-product attention  
> **Companion:** [@MLInterviewMath](https://x.com/MLInterviewMath)

## Interview question

Scaled dot-product attention uses

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Why is the denominator specifically $\sqrt{d_k}$?

Assume the coordinates of a query $q\in\mathbb{R}^{d_k}$ and key $k\in\mathbb{R}^{d_k}$ are independent, zero-mean, and unit-variance.

---

## 1. Start from one attention logit

For one query-key pair,

$$
s=q^\top k=\sum_{i=1}^{d_k}q_i k_i.
$$

Under the stated assumptions,

$$
\mathbb{E}[q_i k_i]
=
\mathbb{E}[q_i]\mathbb{E}[k_i]
=
0.
$$

Also,

$$
\operatorname{Var}(q_i k_i)
=
\mathbb{E}[q_i^2k_i^2]
=
\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]
=
1.
$$

Therefore

$$
\mathbb{E}[s]=0,
\qquad
\operatorname{Var}(s)
=
\sum_{i=1}^{d_k}\operatorname{Var}(q_i k_i)
=
d_k.
$$

So

$$
\operatorname{Std}(s)=\sqrt{d_k}.
$$

The raw attention logit therefore acquires a dimension-dependent scale even though the coordinate-level statistics are unchanged.

---

## 2. What the scaling is controlling

Now define

$$
\tilde{s}=\frac{s}{\sqrt{d_k}}.
$$

Then

$$
\operatorname{Var}(\tilde{s})
=
\frac{\operatorname{Var}(s)}{d_k}
=
1.
$$

The $1/\sqrt{d_k}$ factor removes the growth in logit variance caused purely by summing more coordinate-wise products.

That matters because these logits are immediately passed through softmax.

---

## 3. Why larger logits change softmax behavior

For

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},
$$

the softmax Jacobian is

$$
\frac{\partial p_i}{\partial z_j}
=
p_i(\delta_{ij}-p_j).
$$

As the spread of the logits grows, softmax tends to become sharper. When one probability is close to 1 and the others are close to 0, most entries of the Jacobian become small.

So the point of the scaling is more precise than “numerical stability”: it prevents the softmax temperature from changing systematically just because $d_k$ changed.

---

## 4. Why not divide by dₖ?

If instead we used

$$
\frac{s}{d_k},
$$

then

$$
\operatorname{Var}\!\left(\frac{s}{d_k}\right)
=
\frac{1}{d_k}.
$$

As $d_k$ grows, the logits shrink toward zero and the softmax approaches the uniform distribution.

| Scaling | Logit variance | Tendency as dₖ grows |
|---|---:|---|
| none | $d_k$ | sharper / more saturated |
| $1/\sqrt{d_k}$ | $1$ | dimension-stable scale |
| $1/d_k$ | $1/d_k$ | increasingly uniform |

The square-root factor is exactly what removes the linear growth in variance.

---

## 5. What assumption is hidden in the textbook derivation?

Unit variance is convenient, not fundamental.

Suppose instead

$$
\operatorname{Var}(q_i)=\sigma_q^2,
\qquad
\operatorname{Var}(k_i)=\sigma_k^2,
$$

with zero means and independent coordinates. Then

$$
\operatorname{Var}(q_i k_i)
=
\sigma_q^2\sigma_k^2,
$$

so

$$
\operatorname{Var}(q^\top k)
=
d_k\sigma_q^2\sigma_k^2.
$$

After the usual scaling,

$$
\operatorname{Var}\!\left(
\frac{q^\top k}{\sqrt{d_k}}
\right)
=
\sigma_q^2\sigma_k^2.
$$

So $1/\sqrt{d_k}$ removes **growth with dimensionality**. It does **not** guarantee unit-variance logits under arbitrary query/key statistics.

That is a useful interview follow-up because it separates the dimensional argument from assumptions about activation scale.

---

## 6. What if coordinates are correlated?

Let $X_i=q_i k_i$. In general,

$$
\operatorname{Var}\!\left(\sum_iX_i\right)
=
\sum_i\operatorname{Var}(X_i)
+
2\sum_{i<j}\operatorname{Cov}(X_i,X_j).
$$

The clean $\operatorname{Var}(s)=d_k$ result assumes those covariance terms vanish.

Real Transformer activations are not literally iid. The standard derivation is a scale argument that explains the architectural normalization, not an exact probabilistic model of every trained layer.

---

## 7. Interview follow-ups

### Why scale before softmax rather than after it?

Scaling the logits changes the distribution produced by softmax. Scaling the probabilities afterward neither reverses saturation nor preserves their normalization in the same way.

### What role do LayerNorm/RMSNorm and initialization play?

They help control the statistics entering the Q/K projections. The $1/\sqrt{d_k}$ factor addresses a different source of growth: accumulation across $d_k$ coordinates.

### What would you measure experimentally?

A useful diagnostic is normalized attention entropy as $d_k$ changes.

Under the simple iid model:

- no scaling should make entropy fall with $d_k$;
- $1/\sqrt{d_k}$ should keep it comparatively stable;
- $1/d_k$ should push entropy toward the uniform maximum.

Run the experiment in [code/001_attention_scaling.py](../code/001_attention_scaling.py).

---

## 8. What a strong interview answer should cover

A concise but complete answer should be able to reconstruct:

- $\operatorname{Var}(q^\top k)=d_k$ under the stated assumptions;
- why $1/\sqrt{d_k}$ removes dimension dependence;
- what happens to softmax with no scaling;
- why $1/d_k$ over-corrects;
- which assumptions the derivation depends on.

---

## From *Underneath the Surface*

This question is adapted from the **Transformer Mechanics** chapter of *Underneath the Surface: First-Principles ML & LLM Interview Prep for Frontier Labs*.

The book develops the surrounding topics as one connected sequence:

**attention math → normalization and FFNs → RoPE → KV cache → GQA/MQA → prefill/decode → FlashAttention → MoE**

[About the book →](../book/about.md)

---

[← Question index](../README.md#start-here) · [Transformer question map →](../topics/transformers.md)

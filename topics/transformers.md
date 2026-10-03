# Transformer Mechanics — Question Map

This sequence starts from one attention logit and works outward toward serving and distributed systems.

## Attention math

1. **[Why does attention divide by √dₖ?](../questions/001-attention-scaling.md)**  
   Variance scaling, softmax saturation, and assumptions behind the standard derivation.

2. **When does the N² attention term dominate?**  
   MHA parameter count, projection FLOPs, attention FLOPs, and compute regimes.

3. **Why is causal training parallel?**  
   Lower-triangular masks, factorized likelihood, and the sequential decode dependency.

## Block mechanics

4. **How does RMSNorm change the backward geometry?**  
   LayerNorm vs RMSNorm forward/backward and the mean-direction term.

5. **Why does SwiGLU use a hidden width near 8d/3?**  
   Parameter-budget derivation, gating, and parameter-count vs runtime trade-offs.

## Position

6. **Why does RoPE produce relative-position-dependent scores?**  
   Rotation algebra, relative offset, and extrapolation vs representability.

## Serving

7. **How large is the KV cache?**  
   Cache memory, concurrency, context length, and MHA/GQA/MQA.

8. **How can FlashAttention compute exact softmax block-wise?**  
   Online normalization, HBM traffic, and IO complexity.

## Sparse models

9. **Where does MoE move the bottleneck?**  
   Top-k routing, active vs total parameters, load balance, and all-to-all communication.

---

The sequence is cumulative: later questions reuse quantities and reasoning developed earlier rather than treating each topic as an isolated definition.

[← Repository home](../README.md)

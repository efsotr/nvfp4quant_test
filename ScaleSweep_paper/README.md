# ScaleSweep

For the full paper, see [ScaleSweep.pdf](./ScaleSweep.pdf).

## Core Idea 

The core idea of `ScaleSweep` is to select an FP8 E4M3 scale by sweeping a small analytically derived range around the base scale, rather than searching over all FP8 scales.

## Notation

Given a 16-element block $\mathbf{x}=\{x_i\}_{i=0}^{15}$, the base scale is

$$
s_{\text{base}}=\frac{\max_i |x_i|}{6},\qquad s_{\text{base}}^{\text{FP8}}=\lfloor s_{\text{base}}\rfloor_{\text{FP8}}.
$$

The rounded FP8 base scale is

$$
s_{\text{AbsMax}}^{\text{FP8}}=\lfloor s_{\text{base}}\rceil_{\text{FP8}}.
$$

Let $G_{\text{FP4}}$ and $G_{\text{FP8}}$ be the representable value sets of FP4 E2M1 and FP8 E4M3, respectively. For a format $F$, $\lfloor x\rceil_F$ denotes rounding to the nearest value in $G_F$, and $\lfloor x\rfloor_F$ denotes rounding downward to the largest value in $G_F$ not exceeding $x$. Let $I_{\text{E4M3}}(\cdot)$ denote the integer interpretation of an FP8 E4M3 bit pattern.

**We assume that the local neighborhoods around the base scale are valid positive finite FP8 scales. Therefore, the sweep ranges below do not require FP8 range clipping.**

## ScaleSweep_MSE

`ScaleSweep_MSE` targets the unweighted FP4 reconstruction loss (MSE loss)

$$
\mathcal{L}(s;x)=\sum_{i=0}^{15}\left(x_i-\left\lfloor \frac{x_i}{s}\right\rceil_{\text{FP4}}\cdot s\right)^2.
$$

Let the optimal FP8 scale under MSE be

$$
s_{\text{MSE}}^{\star}=\arg\min_{s\in G_{\text{FP8}}}\mathcal{L}(s;x).
$$

For a 16-element block, the MSE derivation in [ScaleSweep.pdf](./ScaleSweep.pdf) gives the local bit-pattern bound

$$
I_{\text{E4M3}}\left(s_{\text{MSE}}^{\star}\right)\in
\left[
I_{\text{E4M3}}\left(s_{\text{base}}^{\text{FP8}}\right)-3,\
I_{\text{E4M3}}\left(s_{\text{base}}^{\text{FP8}}\right)+7
\right].
$$

Therefore, `ScaleSweep_MSE` selects

$$
s_{\mathrm{ScaleSweep\_MSE}}=
\arg\min_{\substack{
s\in G_{\text{FP8}},\
I_{\text{E4M3}}(s)\in
\left[
I_{\text{E4M3}}\left(s_{\text{base}}^{\text{FP8}}\right)-3,\
I_{\text{E4M3}}\left(s_{\text{base}}^{\text{FP8}}\right)+7
\right]
}}
\mathcal{L}(s;x).
$$

> **Note:** The rounded FP8 base-scale sweep range below is not included in the ScaleSweep paper. It is introduced only for implementation convenience.

When using the rounded FP8 base scale, one can also derive that the same MSE optimum satisfies

$$
I_{\text{E4M3}}\left(s_{\text{MSE}}^{\star}\right)\in
\left[
I_{\text{E4M3}}\left(s_{\text{AbsMax}}^{\text{FP8}}\right)-3,\
I_{\text{E4M3}}\left(s_{\text{AbsMax}}^{\text{FP8}}\right)+7
\right].
$$

Therefore, `ScaleSweep_MSE_round` selects

$$
s_{\mathrm{ScaleSweep\_MSE\_round}}=
\arg\min_{\substack{
s\in G_{\text{FP8}},\
I_{\text{E4M3}}(s)\in
\left[
I_{\text{E4M3}}\left(s_{\text{AbsMax}}^{\text{FP8}}\right)-3,\
I_{\text{E4M3}}\left(s_{\text{AbsMax}}^{\text{FP8}}\right)+7
\right]
}}
\mathcal{L}(s;x).
$$

## ScaleSweep

`ScaleSweep` targets the weighted FP4 reconstruction loss (weighted MSE loss)

$$
\mathcal{L}(s;x,w)=
\sum_{i=0}^{15}w_i
\left(x_i-\left\lfloor \frac{x_i}{s}\right\rceil_{\text{FP4}}\cdot s\right)^2,
\qquad w_i\ge 0.
$$

Let the optimal FP8 scale under WMSE be

$$
s_{\text{WMSE}}^{\star}=\arg\min_{s\in G_{\text{FP8}}}\mathcal{L}(s;x,w).
$$

For the weighted objective, the WMSE derivation in [ScaleSweep.pdf](./ScaleSweep.pdf) preserves the upper bound

$$
I_{\text{E4M3}}\left(s_{\text{WMSE}}^{\star}\right)
\le
I_{\text{E4M3}}\left(s_{\text{base}}^{\text{FP8}}\right)+7,
$$

while the lower bound can reach the smallest positive FP8 value. Therefore, `ScaleSweep` approximately uses $s_{\text{base}}/2$ as the lower bound and selects

$$
s_{\mathrm{ScaleSweep}}=
\arg\min_{\substack{
s\in G_{\text{FP8}},\
I_{\text{E4M3}}(s)\in
\left[
I_{\text{E4M3}}\left(s_{\text{base}}^{\text{FP8}}\right)-8,\
I_{\text{E4M3}}\left(s_{\text{base}}^{\text{FP8}}\right)+7
\right]
}}
\mathcal{L}(s;x,w).
$$

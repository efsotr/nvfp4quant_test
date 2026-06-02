# ScaleSweep

For the full paper, see [ScaleSweep.pdf](./ScaleSweep.pdf).

## Core Method Summary

`ScaleSweep_MSE` and `ScaleSweep` choose an FP8 E4M3 scale for FP4 E2M1 block quantization. Given a 16-element block $x=\{x_i\}_{i=0}^{15}$, the base scale is

$$s_{\mathrm{base}}=\frac{\max_i |x_i|}{6},\qquad s_{\mathrm{base}}^{\mathrm{FP8}}=\lfloor s_{\mathrm{base}}\rfloor_{\mathrm{FP8}}.$$

The rounded FP8 base scale is

$$s_{\mathrm{AbsMax}}^{\mathrm{FP8}}=\lfloor s_{\mathrm{base}}\rceil_{\mathrm{FP8}}.$$

Let $G_{\mathrm{FP4}}$ and $G_{\mathrm{FP8}}$ be the representable value sets of FP4 E2M1 and FP8 E4M3, respectively. For a format $F$, $\lfloor x\rceil_F$ denotes rounding to the nearest value in $G_F$, and $\lfloor x\rfloor_F$ denotes rounding downward to the largest value in $G_F$ not exceeding $x$. Let $I_{\mathrm{E4M3}}(\cdot)$ denote the integer interpretation of an FP8 E4M3 bit pattern.

**We assume that the local neighborhoods around the base scale are valid positive finite FP8 scales. Therefore, the sweep ranges below do not require FP8 range clipping.**

## ScaleSweep_MSE

`ScaleSweep_MSE` targets the unweighted FP4 reconstruction loss

$$\mathcal{L}(s;x)=\sum_{i=0}^{15}\left(x_i-\left\lfloor \frac{x_i}{s}\right\rceil_{\mathrm{FP4}}\cdot s\right)^2.$$

Let the optimal FP8 scale under MSE be

$$s_{\mathrm{MSE}}^\star=\arg\min_{s\in G_{\mathrm{FP8}}}\mathcal{L}(s;x).$$

For a 16-element block, the MSE derivation in [ScaleSweep.pdf](./ScaleSweep.pdf) gives the local bit-pattern bound

$$I_{\mathrm{E4M3}}(s_{\mathrm{MSE}}^\star)\in \left[I_{\mathrm{E4M3}}\!\left(s_{\mathrm{base}}^{\mathrm{FP8}}\right)-3,\ I_{\mathrm{E4M3}}\!\left(s_{\mathrm{base}}^{\mathrm{FP8}}\right)+7\right].$$

Therefore, `ScaleSweep_MSE` selects

$$s_{\mathrm{ScaleSweep\_MSE}}=\arg\min_{\substack{s\in G_{\mathrm{FP8}},\ I_{\mathrm{E4M3}}(s)\in [I_{\mathrm{E4M3}}(s_{\mathrm{base}}^{\mathrm{FP8}})-3,\ I_{\mathrm{E4M3}}(s_{\mathrm{base}}^{\mathrm{FP8}})+7]}}\mathcal{L}(s;x).$$

> **Note:** The rounded FP8 base-scale sweep range below is not included in the ScaleSweep paper. It is introduced only for implementation convenience.

When using the rounded FP8 base scale, one can also derive that the same MSE optimum satisfies

$$I_{\mathrm{E4M3}}(s_{\mathrm{MSE}}^\star)\in \left[I_{\mathrm{E4M3}}\!\left(s_{\mathrm{AbsMax}}^{\mathrm{FP8}}\right)-3,\ I_{\mathrm{E4M3}}\!\left(s_{\mathrm{AbsMax}}^{\mathrm{FP8}}\right)+7\right].$$

Therefore, `ScaleSweep_MSE_round` selects

$$s_{\mathrm{ScaleSweep\_MSE\_round}}=\arg\min_{\substack{s\in G_{\mathrm{FP8}},\ I_{\mathrm{E4M3}}(s)\in [I_{\mathrm{E4M3}}(s_{\mathrm{AbsMax}}^{\mathrm{FP8}})-3,\ I_{\mathrm{E4M3}}(s_{\mathrm{AbsMax}}^{\mathrm{FP8}})+7]}}\mathcal{L}(s;x).$$

## ScaleSweep

`ScaleSweep` targets the weighted FP4 reconstruction loss

$$\mathcal{L}(s;x,w)=\sum_{i=0}^{15}w_i\left(x_i-\left\lfloor \frac{x_i}{s}\right\rceil_{\mathrm{FP4}}\cdot s\right)^2,\qquad w_i\ge 0.$$

Let the optimal FP8 scale under WMSE be

$$s_{\mathrm{WMSE}}^\star=\arg\min_{s\in G_{\mathrm{FP8}}}\mathcal{L}(s;x,w).$$

For the weighted objective, the WMSE derivation in [ScaleSweep.pdf](./ScaleSweep.pdf) preserves the upper bound

$$I_{\mathrm{E4M3}}(s_{\mathrm{WMSE}}^\star)\le I_{\mathrm{E4M3}}\!\left(s_{\mathrm{base}}^{\mathrm{FP8}}\right)+7,$$

while the lower bound can reach the smallest positive FP8 value. Therefore, `ScaleSweep` approximately uses $s_{\mathrm{base}}/2$ as the lower bound and selects

$$s_{\mathrm{ScaleSweep}}=\arg\min_{\substack{s\in G_{\mathrm{FP8}},\ I_{\mathrm{E4M3}}(s)\in [I_{\mathrm{E4M3}}(s_{\mathrm{base}}^{\mathrm{FP8}})-8,\ I_{\mathrm{E4M3}}(s_{\mathrm{base}}^{\mathrm{FP8}})+7]}}\mathcal{L}(s;x,w).$$
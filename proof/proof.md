# Proof of the NVFP4 ScaleSweep Range

This note proves the FP8 E4M3 block-scale search range used by ScaleSweep for NVFP4 under the MSE objective. The weighted case keeps the same upper bound, but does not have a universal lower bound.

## Notation

Consider one 16-element micro-block $\mathbf{x}=\{x_i\}_{i=0}^{15}$. Let

$$
M=\max_i |x_i|,\qquad s_{\mathrm{base}}=\frac{M}{6}.
$$

Let

$$
q_0=\left\lfloor s_{\mathrm{base}}\right\rceil_{\mathrm{FP8}}
$$

be the nearest positive finite FP8 E4M3 scale. Write $\mathrm{bits}(q_0)$ for its positive sign-0 FP8 bit-pattern.

The nonnegative FP4 E2M1 codebook is

$$
\mathcal{G}=\{0,\tfrac12,1,\tfrac32,2,3,4,6\}.
$$

For scale $s>0$, define

$$
d_s(x)=\min_{g\in\mathcal{G}_{\mathrm{FP4}}}(x-sg)^2.
$$

By symmetry of the FP4 grid, the MSE proof may use $|x_i|$ and the nonnegative codebook $\mathcal{G}$.

The MSE objective is

$$
\mathcal{L}(s;\mathbf{x}) =
\sum_{i=0}^{15}
\left(x_i-\left\lfloor x_i/s\right\rceil_{\mathrm{FP4}}\cdot s\right)^2.
$$

Let

$$
s^\star\in
\arg\min_{s\in\mathcal{G}_{\mathrm{FP8}}^+}
\mathcal{L}(s;\mathbf{x}).
$$

The goal is to prove that one MSE minimizer satisfies

$$
\mathrm{bits}(s^\star)
\in
[\mathrm{bits}(q_0)-3,\ \mathrm{bits}(q_0)+7],
$$

away from FP8 saturation.

## Upper Bound

If

$$
s\ge \frac{M}{3.5},
$$

then

$$
\mathcal{L}(s;\mathbf{x})\ge \mathcal{L}(s/2;\mathbf{x}).
$$

Indeed, $|x_i|\le 3.5s$ for every $i$. At scale $s$, the FP4 decision boundary between $3s$ and $4s$ is $3.5s$. Hence no loss-relevant reconstruction above $3s$ is needed. The used reconstruction set is contained in the FP4 grid at scale $s/2$, so nearest-point optimality gives the inequality.

Since $s_{\mathrm{base}}=M/6$, every useful scale may be chosen with

$$
s\le \frac{12}{7}s_{\mathrm{base}}.
$$

Now center at

$$
q_0=\left\lfloor s_{\mathrm{base}}\right\rceil_{\mathrm{FP8}}.
$$

For a normal FP8 value with mantissa field $b$, write its significand as $m/8$, where $m=b+8$. The right endpoint of the nearest-rounding cell is the midpoint between $q_0$ and the next FP8 value. Therefore the largest upper candidate is the largest FP8 value strictly below

$$
\frac{12}{7}\cdot
\frac{q_0+\mathrm{next}(q_0)}{2}.
$$

Enumerating FP8 E4M3 gives the following upper offsets.

### Normal E4M3 upper offsets

|         $b$ |     0 |      1 |     2 |       3 |     4 |       5 |      6 |       7 |
| ----------: | ----: | -----: | ----: | ------: | ----: | ------: | -----: | ------: |
|         $m$ |     8 |      9 |    10 |      11 |    12 |      13 |     14 |      15 |
| upper ratio | $7/4$ | $16/9$ | $8/5$ | $18/11$ | $5/3$ | $22/13$ | $12/7$ | $26/15$ |
|      offset |     6 |      7 |     6 |       6 |     6 |       6 |      6 |       6 |

### Subnormal E4M3 upper offsets

For subnormal scales, write the value as $b2^{-9}$.

|         $b$ |   1 |   2 |     3 |     4 |     5 |      6 |      7 |
| ----------: | --: | --: | ----: | ----: | ----: | -----: | -----: |
| upper ratio | $2$ | $2$ | $5/3$ | $7/4$ | $9/5$ | $11/6$ | $12/7$ |
|      offset |   1 |   2 |     2 |     3 |     4 |      5 |      5 |

Thus the uniform upper offset is $+7$.

The same upper proof applies to WMSE with arbitrary nonnegative weights, because the pointwise inequality is preserved after multiplying by $w_i\ge0$ and summing.

## Lower Bound for MSE

Normalize by $q_0$:

$$
y_i=\frac{|x_i|}{q_0},\qquad r=\frac{s}{q_0}.
$$

For $y\ge0$, define

$$
d_r(y)=\min_{g\in\mathcal{G}}(y-rg)^2.
$$

Up to the positive factor $q_0^2$, minimizing over $s$ is equivalent to minimizing

$$
\overline{\mathcal{L}}(r;\mathbf{y}) = \sum_{i=0}^{15}d_r(y_i).
$$

At least one coordinate is maximal. For such a coordinate,

$$
y_{\max}=6\alpha,
\qquad
\alpha=\frac{s_{\mathrm{base}}}{q_0}.
$$

Since $q_0$ is the nearest FP8 value to $s_{\mathrm{base}}$, $\alpha$ lies in the corresponding nearest-rounding cell. Let its left endpoint be $\alpha_{\min}$.

### Analytic doubling exclusion

Compare a candidate ratio $r$ with $2r$.

For every nonmaximal coordinate,

$$
d_{2r}(y)-d_r(y)\le \frac{r^2}{4}.
$$

For the maximal coordinate,

$$
d_{2r}(6\alpha)-d_r(6\alpha)
\le
(6\alpha-8r)^2-(6\alpha-6r)^2 = -24\alpha r+28r^2.
$$

Hence

$$
\overline{\mathcal{L}}(2r;\mathbf{y}) - \overline{\mathcal{L}}(r;\mathbf{y})
\le
-24\alpha r+\frac{127}{4}r^2.
$$

This is negative whenever

$$
r<\frac{96}{127}\alpha.
$$

Using the worst case $\alpha\ge\alpha_{\min}$, every legal candidate with

$$
r<\frac{96}{127}\alpha_{\min}
$$

is excluded.

### Finite exclusion above the analytic threshold

Only finitely many FP8 ratios remain between the analytic threshold and the claimed lower bound. These are excluded by exact convex dominance.

For a candidate $r$, choose legal ratios $u_j>r$ and weights $\lambda_j\ge0$ with $\sum_j\lambda_j=1$. Define

$$
\Gamma_r(y) = d_r(y)-\sum_j\lambda_j d_{u_j}(y).
$$

Let

$$
A=\inf_{6\alpha_{\min}\le y\le 6\alpha_{\max}}\Gamma_r(y),
\qquad
B=\inf_{0\le y\le 6\alpha_{\max}}\Gamma_r(y).
$$

If

$$
A+15B>0,
$$

then

$$
\overline{\mathcal{L}}(r;\mathbf{y})

>

\sum_j\lambda_j
\overline{\mathcal{L}}(u_j;\mathbf{y}).
$$

Therefore at least one $u_j$ has smaller block loss than $r$, so $r$ cannot be optimal.

The verification is exact: on every FP4 decision cell, $\Gamma_r(y)$ is affine because the $y^2$ terms cancel under $\sum_j\lambda_j=1$. Therefore the infima above are attained at finitely many cell endpoints.

The resulting certified lower offsets are as follows.

### Normal E4M3 lower offsets

|         $b$ |       0 |     1 |     2 |      3 |     4 |       5 |       6 |     7 |
| ----------: | ------: | ----: | ----: | -----: | ----: | ------: | ------: | ----: |
|         $m$ |       8 |     9 |    10 |     11 |    12 |      13 |      14 |    15 |
| lower ratio | $13/16$ | $7/9$ | $4/5$ | $9/11$ | $5/6$ | $10/13$ | $11/14$ | $4/5$ |
|      offset |      -3 |    -3 |    -2 |     -2 |    -2 |      -3 |      -3 |    -3 |

### Subnormal E4M3 lower offsets

|         $b$ |   1 |     2 |     3 |     4 |     5 |     6 |     7 |
| ----------: | --: | ----: | ----: | ----: | ----: | ----: | ----: |
| lower ratio | $1$ | $1/2$ | $1/3$ | $1/4$ | $2/5$ | $1/2$ | $4/7$ |
|      offset |   0 |    -1 |    -2 |    -3 |    -3 |    -3 |    -3 |

Thus the uniform MSE lower offset is $-3$.

## Final Range

Combining the upper and lower bounds, one MSE minimizer satisfies

$$
\mathrm{bits}(s^\star)
\in
[\mathrm{bits}(q_0)-3,\ \mathrm{bits}(q_0)+7],
\qquad
q_0=\left\lfloor \frac{\max_i |x_i|}{6}\right\rceil_{\mathrm{FP8}}.
$$

For WMSE with arbitrary nonnegative weights, the upper bound still holds:

$$
\mathrm{bits}(s^\star_{\mathrm{WMSE}})
\le
\mathrm{bits}(q_0)+7.
$$

No nontrivial universal lower bound exists for arbitrary nonnegative weights, because the maximal entry may have arbitrarily small weight and the optimum may be determined by smaller high-weight entries.

The listed offsets are tight for the enumerated FP8 E4M3 cases: each normal and subnormal boundary has an explicit 16-element example whose verified optimum occurs exactly at the claimed offset.

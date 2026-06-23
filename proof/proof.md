# Proof of the NVFP4 ScaleSweep Range

This file proves the FP8 E4M3 block-scale search ranges used by
ScaleSweep for NVFP4 under MSE and WMSE objectives.

## Notation

Let $[N]=\{0,1,\ldots,N-1\}$.  This proof concerns one 16-element
micro-block
$$\mathbf{x}=\{x_i\}_{i=0}^{15}.$$
The FP4 E2M1 value set is
$$\mathcal{G}_{\mathrm{FP4}}
=\{0,\pm\tfrac12,\pm1,\pm\tfrac32,\pm2,\pm3,\pm4,\pm6\}.$$
Its nonnegative part is
$$\mathcal{G}=\{0,\tfrac12,1,\tfrac32,2,3,4,6\}.$$
The FP8 E4M3 value set is denoted by
$\mathcal{G}_{\mathrm{FP8}}$.  Its positive finite part is denoted by
$\mathcal{G}_{\mathrm{FP8}}^+$.

For a numeric format $\mathcal{F}$, the notation
$$\left\lfloor x\right\rceil_{\mathcal{F}}$$
denotes a nearest element of $\mathcal{G}_{\mathcal{F}}$.  The notation
$$\left\lfloor x\right\rfloor_{\mathcal{F}}$$
denotes the largest element of $\mathcal{G}_{\mathcal{F}}$ not exceeding
$x$, and
$$\left\lceil x\right\rceil_{\mathcal{F}}$$
denotes the smallest element of $\mathcal{G}_{\mathcal{F}}$ not smaller
than $x$.

For $s>0$, define the scaled FP4 format
$$\mathrm{FP4}(s)=\{sg:g\in\mathcal{G}_{\mathrm{FP4}}\}.$$
The weighted FP4 block loss is
$$\mathcal{L}(s;\mathbf{x},\mathbf{w})
=\sum_{i=0}^{15}w_i
\left(x_i-\left\lfloor x_i\right\rceil_{\mathrm{FP4}(s)}\right)^2,$$
where $w_i\ge0$.  The MSE loss is the unweighted case
$$\mathcal{L}(s;\mathbf{x})
=\sum_{i=0}^{15}
\left(x_i-\left\lfloor x_i\right\rceil_{\mathrm{FP4}(s)}\right)^2.$$

Let
$$M_x=\max_i |x_i|,\qquad s_{\mathrm{base}}=\frac{M_x}{6}.$$
The downward FP8 base scale and the nearest FP8 base scale are
$$s_0=\left\lfloor s_{\mathrm{base}}\right\rfloor_{\mathrm{FP8}},
\qquad
s_{\mathrm{round}}
=\left\lfloor s_{\mathrm{base}}\right\rceil_{\mathrm{FP8}}.$$
Let $I_{\mathrm{E4M3}}(s)$ be the unsigned integer interpretation of
the positive FP8 E4M3 bit pattern representing $s$.

The FP4 and FP8 grids are symmetric.  Therefore replacing $x_i$ by
$-x_i$ changes the selected reconstruction value by the same sign, up
to ties that have identical squared error.  Hence the loss contribution
of $x_i$ and $-x_i$ is identical.  In all lower-bound arguments it is
sufficient to consider $x_i\ge0$ and to use $\mathcal{G}$.

### Main Theorem for MSE

Let
$$s_{\mathrm{MSE}}^\star\in
\arg\min_{s\in\mathcal{G}_{\mathrm{FP8}}^+}
\mathcal{L}(s;\mathbf{x}).$$
Assume the local neighborhood is positive finite and not saturated,
specifically $11s_0/7\le448$.  Then one MSE minimizer satisfies
$$I_{\mathrm{E4M3}}(s_{\mathrm{MSE}}^\star)\in
\left[I_{\mathrm{E4M3}}(s_0)-3,\,
I_{\mathrm{E4M3}}(s_0)+7\right].$$
If the search is centered at $s_{\mathrm{round}}$, the same uniform
offset range $[-3,7]$ is valid.

### Main Theorem for WMSE

Let
$$s_{\mathrm{WMSE}}^\star\in
\arg\min_{s\in\mathcal{G}_{\mathrm{FP8}}^+}
\mathcal{L}(s;\mathbf{x},\mathbf{w}).$$
For arbitrary nonnegative weights,
$$I_{\mathrm{E4M3}}(s_{\mathrm{WMSE}}^\star)
\le I_{\mathrm{E4M3}}(s_0)+7.$$
There is no nontrivial universal lower bound depending only on
$s_{\mathrm{base}}$.  If the maximal entry has arbitrarily small weight,
the weighted optimum may be determined by much smaller high-weight
entries instead.

## FP8 Scale Upper Bound

### Scale-Space Bound

If
$$s\ge\frac{M_x}{3.5},$$
then
$$\mathcal{L}(s;\mathbf{x},\mathbf{w})
\ge\mathcal{L}(s/2;\mathbf{x},\mathbf{w}).$$
Indeed, $|x_i|\le3.5s$ for all $i$.  Under $\mathrm{FP4}(s)$, the
decision boundary between $3s$ and $4s$ is $3.5s$.  At equality, for
the purpose of loss comparison, one may take
$$\left\lfloor 3.5s\right\rceil_{\mathrm{FP4}(s)}=3s,$$
since $3s$ and $4s$ give the same squared error.  Thus no loss-relevant
reconstruction value outside
$$\{0,\pm\tfrac12s,\pm s,\pm\tfrac32s,\pm2s,\pm3s\}$$
is needed at scale $s$.  This set is contained in $\mathrm{FP4}(s/2)$.
Therefore each reconstruction value used at scale $s$ is feasible at
scale $s/2$, and nearest-point optimality gives the inequality after
multiplying by $w_i\ge0$ and summing.

Since $s_{\mathrm{base}}=M_x/6$, every useful scale may be chosen with
$$s\le\frac{12}{7}s_{\mathrm{base}}.$$

### Bit-Pattern Tables

The bit-pattern upper offsets are obtained by enumerating FP8 E4M3
values.  If $I=I_{\mathrm{E4M3}}(s_0)$, then
$s_{\mathrm{base}}$ lies below the next FP8 value.  Hence the largest
possible upper candidate is the largest FP8 value below
$12/7$ times that next value.  The enumeration is implemented in
[`nvfp4_upper_bound.py`](nvfp4_upper_bound.py).

For normal E4M3 scales, write the significand as $m/8$, where
$m=b+8$ and $b\in\{0,\ldots,7\}$.

| b | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $m$ | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| offset | 6 | 7 | 6 | 6 | 6 | 6 | 6 | 6 |

For subnormal E4M3 scales, write the value as $b2^{-9}$, where
$b\in\{1,\ldots,7\}$.

| b | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| offset | 1 | 2 | 2 | 3 | 4 | 5 | 5 |

Both tables are bounded by the uniform upper offset $+7$.

## FP8 Scale Lower Bound

### Normalized Loss

For the MSE proof, define
$$y_i=\frac{|x_i|}{s_0},\qquad r=\frac{s}{s_0}.$$
For $y\ge0$, define
$$d_r(y)=\min_{g\in\mathcal{G}}(y-rg)^2.$$
Then MSE minimization over $s$ is equivalent, up to the positive factor
$s_0^2$, to minimizing
$$\overline{\mathcal{L}}(r;\mathbf{y})
=\sum_{i=0}^{15}d_r(y_i).$$
At least one coordinate is maximal.  For such a coordinate,
$$y_{\max}=6\alpha,\qquad \alpha=\frac{s_{\mathrm{base}}}{s_0}\ge1.$$

For normal $s_0=(m/8)2^a$, where $m\in\{8,\ldots,15\}$,
$$1\le\alpha<\rho_m,\qquad
\rho_m=\begin{cases}(m+1)/m,&m<15,\\16/15,&m=15.\end{cases}$$
For subnormal $s_0=b2^{-9}$,
$$1\le\alpha<\rho_b,\qquad \rho_b=(b+1)/b.$$

### Analytic Exclusion Below $96/127$

Compare a candidate ratio $r$ with $2r$.  For all nonmaximal
coordinates,
$$d_{2r}(y)-d_r(y)\le\frac{r^2}{4}.$$
For the maximal coordinate,
$$d_{2r}(6\alpha)-d_r(6\alpha)
\le(6\alpha-8r)^2-(6\alpha-6r)^2
=-24\alpha r+28r^2.$$
Therefore
$$\overline{\mathcal{L}}(2r;\mathbf{y})
-\overline{\mathcal{L}}(r;\mathbf{y})
\le -24\alpha r+\frac{127}{4}r^2.$$
This is negative whenever
$$r<\frac{96}{127}\alpha.$$
Since $\alpha\ge1$, every legal candidate with
$$r<\frac{96}{127}$$
is excluded.  This is the strict version of the earlier $3/4$ cutoff.

### Finite Exclusion Above $96/127$

After the analytic exclusion, only a finite number of legal FP8 ratios
below the eventual lower bound remain.  They are excluded by convex
certificates.  For a candidate $r$, choose legal ratios $u_j>r$ and
weights $\lambda_j\ge0$ with $\sum_j\lambda_j=1$.  Define
$$\Gamma_r(y)=d_r(y)-\sum_j\lambda_jd_{u_j}(y).$$
Let
$$A=\inf_{6\le y\le6\rho}\Gamma_r(y),\qquad
B=\inf_{0\le y\le6\rho}\Gamma_r(y).$$
If $A+15B>0$, then one of the $u_j$ has strictly smaller block loss
than $r$, so $r$ is not optimal.

For the downward base $s_0$, the only normal candidates above
$96/127$ that must be excluded are:

| $b$ | $m$ | excluded $r$ | comparison $U$ | margin |
|---:|---:|---:|---|---:|
| 1 | 9 | $7/9$ | $\{5/6,1,14/9\}$ | $100/1539$ |
| 5 | 13 | $10/13$ | $\{14/13,20/13\}$ | $324/10309$ |
| 6 | 14 | $11/14$ | $\{6/7,15/14,11/7\}$ | $244/2695$ |

For subnormal downward bases, all candidates below the lower table are
already below $96/127$; no additional certificate is needed.

The certificate generation and exact rational verification are in
[`nvfp4_lower_bound_lp.py`](nvfp4_lower_bound_lp.py).  The independent
checker and tight examples are in
[`nvfp4_lower_bound.py`](nvfp4_lower_bound.py).  The proof is exact
because $\Gamma_r(y)$ is affine on each FP4 decision cell: the $y^2$
terms cancel due to $\sum_j\lambda_j=1$, so it suffices to enumerate
cell endpoints.

### Downward-Base Lower Bit Tables

The previous exclusions prove
$$s_{\mathrm{MSE}}^\star\ge\frac45s_0.$$
The resulting bit-pattern lower bound is
$$\left\lceil\frac45s_0\right\rceil_{\mathrm{FP8}}.$$

For normal downward bases:

| $b$ | $m$ | lower ratio | offset |
|---:|---:|---:|---:|
| 0 | 8 | $13/16$ | -3 |
| 1 | 9 | $5/6$ | -2 |
| 2 | 10 | $4/5$ | -2 |
| 3 | 11 | $9/11$ | -2 |
| 4 | 12 | $5/6$ | -2 |
| 5 | 13 | $11/13$ | -2 |
| 6 | 14 | $6/7$ | -2 |
| 7 | 15 | $4/5$ | -3 |

For subnormal downward bases:

| $b$ | lower ratio | offset |
|---:|---:|---:|
| 1 | $1$ | 0 |
| 2 | $1$ | 0 |
| 3 | $1$ | 0 |
| 4 | $1$ | 0 |
| 5 | $4/5$ | -1 |
| 6 | $5/6$ | -1 |
| 7 | $6/7$ | -1 |

Hence the uniform downward-base lower offset is $-3$.

### Nearest-Base Lower Bit Tables

Now center the search at
$$s_{\mathrm{round}}
=\left\lfloor s_{\mathrm{base}}\right\rceil_{\mathrm{FP8}}.$$
The same exclusion method applies, but the maximal coordinate satisfies
$$y_{\max}=6\alpha,\qquad
\alpha=\frac{s_{\mathrm{base}}}{s_{\mathrm{round}}}.$$
For each nearest FP8 cell, the analytic exclusion uses
$$r<\frac{96}{127}\alpha_{\min},$$
where $\alpha_{\min}$ is the left endpoint of that cell divided by
$s_{\mathrm{round}}$.  The remaining normal candidates are excluded by
the same convex-certificate method.

For normal nearest bases, the certified lower table is:

| $b$ | $m$ | lower ratio | offset |
|---:|---:|---:|---:|
| 0 | 8 | $13/16$ | -3 |
| 1 | 9 | $7/9$ | -3 |
| 2 | 10 | $4/5$ | -2 |
| 3 | 11 | $9/11$ | -2 |
| 4 | 12 | $5/6$ | -2 |
| 5 | 13 | $10/13$ | -3 |
| 6 | 14 | $11/14$ | -3 |
| 7 | 15 | $4/5$ | -3 |

For subnormal nearest bases, clipping to the positive finite FP8 range
gives:

| $b$ | lower ratio | offset |
|---:|---:|---:|
| 1 | $1$ | 0 |
| 2 | $1/2$ | -1 |
| 3 | $1/3$ | -2 |
| 4 | $1/4$ | -3 |
| 5 | $2/5$ | -3 |
| 6 | $1/2$ | -3 |
| 7 | $4/7$ | -3 |

The normal nearest-base certificates are verified in
[`round_based_lower_bounds`](nvfp4_round_based_lower_bounds_by_b.py).
The special upper-half case is checked in
[`script`](nvfp4_round_upper_half_exclude_floor_minus3.py).
Since every nearest-base offset is at least $-3$, the uniform
nearest-base MSE lower offset is also $-3$.

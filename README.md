# Linear Regression from Scratch

An implementation of univariate and multiple linear regression using only NumPy, with
batch gradient descent, z-score feature normalization, model evaluation.

**No scikit-learn is used for the model itself.**

The goal is to predict the resale price of a used laptop from its specifications.

---

## The Data

The dataset is generated from a known linear formula with a small amount of
random variation added to each price. Because the underlying formula is known, the learned
weights can be checked against the true values **the model is verified, not just evaluated.**

| File                            | Rows | Contents                 |
| ------------------------------- | ---- | ------------------------ |
| `data/laptops_train.csv`        | 400  | 7 features + `price`     |
| `data/laptops_test.csv`         | 100  | 7 features only          |
| `data/laptops_test_answers.csv` | 100  | `price` for the test set |

### Features

| Column           | Range           | Notes                                                  |
| ---------------- | --------------- | ------------------------------------------------------ |
| `ram_gb`         | 4 – 32          | discrete: 4, 8, 16, 32                                 |
| `storage_gb`     | 128 – 1024      | discrete: 128, 256, 512, 1024                          |
| `age_years`      | 0 – 7           | the only feature with a negative relationship to price |
| `screen_in`      | 13.0 – 17.3     | continuous                                             |
| `cpu_score`      | ~2,500 – 30,000 | benchmark score; correlated with age                   |
| `battery_health` | 55 – 100        | percent; degrades with age                             |
| `listing_views`  | 5 – 899         | **irrelevant** — has no effect on price                |

**Target:** `price` in US dollars, roughly $650 – $3,100, mean ≈ $1,716.

Note the scale spread: `screen_in` sits around 15 while `cpu_score` reaches 30,000. Four
orders of magnitude between features is what makes normalization/feature scaling necessary.

### Ground truth

The generating formula, used to validate the learned weights:

```
price = 150
      +  18    * ram_gb
      +   0.25 * storage_gb
      -  45    * age_years
      +  12    * screen_in
      +   0.06 * cpu_score
      +   3    * battery_health
      +   0    * listing_views
      + random variation of about $40 per laptop
```

That random variation is the limit on how well any model can do here. It averages around
$40 per laptop and carries no information that the features could explain, so a perfect
model would still be off by roughly that much. In squared terms it puts a floor of about
**1,600** on the achievable Mean Squared Error (MSE).

---

## Part 1 — Univariate Linear Regression

Model: `f(x) = wx + b`, fit on `cpu_score` alone.

Three functions, all written with explicit loops first for clarity:

- `compute_cost(x, y, w, b)` — squared error cost, `J = (1/2m) Σ (f(xᵢ) - yᵢ)²`
- `compute_gradient(x, y, w, b)` — returns `dj_dw`, `dj_db`
- `gradient_descent(...)` — batch updates, returns final parameters plus `J_history`

**Result:** `cpu_score` alone explains only part of the variation in price. The fitted line
has the right slope but the surrounding cloud of points is wide, which motivates adding the
remaining features.

---

## Part 2 — Multiple Linear Regression

Model: `f(x) = w · x + b`, where `w` is a vector of length `n = 7`.

The cost and gradient functions are re-implemented as `compute_cost_mult` and
`compute_gradient_mult`. The structural change is the prediction line:

```python
f_wb = np.dot(X[i], w) + b     # sums over features, instead of w * x[i] + b
```

`gradient_descent` needs no changes — `w = w - alpha * dj_dw` already operates
element-wise on arrays.

**Index convention:** `i` indexes examples (to `m = 400`), `j` indexes features
(to `n = 7`). `X` has shape `(400, 7)`; `X[i]` is one laptop, `X[:, j]` is one feature
across all laptops. Conflating the two is the single easiest bug to write here.

---

## Part 3 — Feature Scaling and Normalization

Z-score normalization, with statistics computed on the **training set only**:

```python
mu    = np.mean(X_train, axis=0)   # one mean per feature
sigma = np.std(X_train, axis=0)
X_norm = (X_train - mu) / sigma
```

The same `mu` and `sigma` are reused for every prediction, including the test set,
recomputing them on new data is unnecessary.

### Why it is required here

Gradients are proportional to feature magnitude. The `cpu_score` gradient is roughly
2,000× larger than the `screen_in` gradient, so:

- any `alpha` large enough to move the small-scale weights sends the `cpu_score` weight
  divergent, producing overflow to `inf` then `nan` within a handful of iterations
- any `alpha` small enough to keep `cpu_score` stable leaves the other six weights
  crawling — at `alpha = 1e-9` the model is still far from converged after 1,000 iterations

Normalization removes the tradeoff. On scaled features, `alpha = 0.01` converges in a few
hundred iterations.

### Effect on the parameters

The target is **not** normalized. With one target there is no cross-scale imbalance to
correct, and leaving `y` in dollars means `w` and `b` come out in interpretable units.

Weights learned on normalized features are in dollars per standard deviation. To recover
the original units:

```python
w_real = w_norm / sigma
b_real = b_norm - np.sum(w_norm * mu / sigma)
```

`b_norm` is the predicted price at the feature means (≈ $1,716, the mean price), which is
a far more meaningful intercept than the price at all-features-zero.

---

## Part 4 — Visualization

| Plot                                             | Graph                                                 |
| ------------------------------------------------ | ----------------------------------------------------- |
| Data Visualization - each X vs price             | ![Data Visualization](images/orig_data_vis)           |
| Cost vs iteration (log y)                        | ![Cost vs Iteration](images/cost_vs_iteration)        |
| Pre-Norm One Feature Linear Regression           | ![Pre-Norm Lin Reg](images/prenorm_lin_reg)           |
| Normalized One Feature Linear Regression         | ![Norm Lin Reg](images/norm_lin_reg)                  |
| Pre-Norm High Alpha Multiple Linear Regression   | ![Pre-Norm Mult Lin Reg](images/prenorm_mult_lin_reg) |
| Normalized Best Alpha Multiple Linear Regression | ![Norm Mult Lin Reg](images/norm_mult_lin_reg)        |
| 3-D Gradient Descent on Cost Surface             | ![Grad Des on Cost Surface](images/grad_on_cost)      |

---

## Part 5 — Prediction and Evaluation

Test predictions apply the training normalization, then the learned parameters:

```python
pred_test = ((X_test - mu) / sigma) @ w_norm + b_norm
```

### Results

```
Test MSE: 1,594   mean absolute error: $30   within $100: 98%
```

Against a floor of about 1,600, the model has recovered essentially all of the learnable
signal — the remaining error is the random variation built into the data, which no model
could predict.

MSE is reported in squared dollars and is not directly comparable to a price; the square
root, ≈ $40, is the interpretable version.

A useful consistency check: `J_history[-1] * 2` equals the training MSE, since the cost
function carries a factor of ½.

### Learned weights vs ground truth

De-normalized weights recover the generating coefficients, including a near-zero weight on
`listing_views` — the model correctly learns to ignore the irrelevant feature without
being told it is irrelevant.

### Single predictions

```python
def predict_price(ram_gb, storage_gb, age_years, screen_in,
                  cpu_score, battery_health, listing_views=400):
    x = np.array([ram_gb, storage_gb, age_years, screen_in,
                  cpu_score, battery_health, listing_views])
    return ((x - mu) / sigma) @ w_norm + b_norm
```

Feature order must match `X_features`; the model has no column names, only positions.

Stepping one feature while holding the rest fixed is a good sanity check — each additional
year of age should reduce the prediction by about $45.

**Known limitation:** the model extrapolates without complaint. A `cpu_score` of 60,000 is
twice anything in the training data, and the prediction follows the line into a price no
real laptop commands. Linear models never refuse to answer.

---

## Requirements

```
numpy
pandas
matplotlib
```

## Concepts implemented

- Squared error cost function
- Analytic gradients for `w` and `b`
- Batch gradient descent with cost history
- Vectorization with `np.dot` and `@` in place of explicit loops
- Z-score normalization with train-only statistics
- Learning rate selection and divergence diagnosis
- Weight de-normalization for interpretation
- Held-out test evaluation against a known error floor

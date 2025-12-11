import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge


st.set_page_config(
    page_title="Overfitting vs Underfitting",
    page_icon="🧠",
    layout="wide",
)


def make_dataset(n_samples: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y_true = np.sin(3 * X[:, 0]) + 0.5 * X[:, 0]
    y = y_true + rng.normal(0.0, noise, size=n_samples)
    return X, y, y_true


def train_model(degree: int, alpha: float):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=0)),
        ]
    )


def plot_fit(X_train, y_train, X_test, y_test, model):
    grid = np.linspace(-1.2, 1.2, 300).reshape(-1, 1)
    y_grid = np.sin(3 * grid[:, 0]) + 0.5 * grid[:, 0]
    preds = model.predict(grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, y_grid, label="True function", color="#444", linewidth=2)
    ax.plot(grid, preds, label="Model prediction", color="#1f77b4", linewidth=2)
    ax.scatter(X_train, y_train, label="Train", color="#2ca02c", alpha=0.7)
    ax.scatter(X_test, y_test, label="Validation", color="#d62728", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Model fit vs ground truth")
    ax.legend()
    ax.grid(True, alpha=0.2)
    return fig


st.title("Overfitting vs Underfitting")
st.write(
    "Adjust complexity and regularization to see how bias/variance changes. "
    "We fit a polynomial model to noisy data drawn from a known target curve."
)

with st.sidebar:
    st.header("Data")
    seed = st.slider("Random seed", 0, 10_000, 1234)
    n_samples = st.slider("Samples", 30, 300, 80, step=10)
    noise = st.slider("Noise std", 0.0, 1.0, 0.2, step=0.05)

    st.header("Model")
    degree = st.slider("Polynomial degree", 1, 20, 10)
    alpha = st.slider(
        "Regularization strength (lambda)",
        0.0,
        20.0,
        1.0,
        step=0.1,
        help="Higher values penalize large weights and reduce variance.",
    )
    train_frac = st.slider("Train split", 0.4, 0.9, 0.7)

X, y, _ = make_dataset(n_samples, noise, seed)
split_idx = int(train_frac * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

model = train_model(degree, alpha)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

# simple diagnosis for under/overfitting
gap = test_mse - train_mse
if test_mse > 2 * train_mse and gap > 1e-6:
    status = "Overfitting"
    suggestion = "Increase `alpha` (regularization), lower `degree`, or add more training data."
elif train_mse > 0.1 and test_mse > 0.1 and abs(gap) < 0.2 * train_mse:
    status = "Underfitting"
    suggestion = "Increase `degree` (model complexity) or reduce `alpha` to add capacity."
else:
    status = "Good fit"
    suggestion = "Model looks balanced — small tuning may help."

col_left, col_right = st.columns([2, 1], gap="large")
with col_left:
    fig = plot_fit(X_train, y_train, X_test, y_test, model)
    st.pyplot(fig, use_container_width=True)

with col_right:
    st.subheader("Metrics")
    st.metric("Train MSE", f"{train_mse:.4f}")
    st.metric("Validation MSE", f"{test_mse:.4f}")
    st.metric("Gap (Val - Train)", f"{gap:.4f}")
    st.caption(
        "Low degree with high bias underfits (both errors high). "
        "Very high degree with low regularization can overfit (train low, val high). "
        "Increasing lambda usually smooths the curve and improves validation error."
    )

st.divider()

st.subheader("📚 Understanding Overfitting vs Underfitting")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 **Underfitting**")
    st.write("""
    **What to look for:**
    - ❌ **Train MSE HIGH** (model can't fit data)
    - ❌ **Validation MSE HIGH** (poor generalization)
    - 📊 Prediction curve is too simple/flat
    
    **Cause:**
    - Degree too low (not enough capacity)
    - Alpha too high (over-regularized)
    
    **Solution:**
    - ⬆️ Increase `Polynomial degree`
    - ⬇️ Decrease `Regularization strength (lambda)`
    """)

with col2:
    st.markdown("### 🔴 **Overfitting**")
    st.write("""
    **What to look for:**
    - ✅ **Train MSE LOW** (fits training data well)
    - ❌ **Validation MSE HIGH** (poor on new data)
    - 📊 Large gap between train and validation error
    - 🌀 Prediction curve wiggles to fit noise
    
    **Cause:**
    - Degree too high (too much capacity)
    - Alpha too low (under-regularized)
    
    **Solution:**
    - ⬇️ Decrease `Polynomial degree`
    - ⬆️ Increase `Regularization strength (lambda)`
    """)

st.divider()

st.subheader("🔧 How Regularization (Lambda) Works")
st.write("""
**Regularization** is the `lambda (λ)` parameter that penalizes large coefficients:

- **λ = 0**: No regularization → model can overfit (memorizes noise)
- **λ = low (0.01-0.1)**: Weak penalty → allows some overfitting
- **λ = medium (0.5-2.0)**: Good balance → typically best generalization
- **λ = high (10+)**: Strong penalty → model becomes too simple (underfitting)

**Try these experiments:**
1. Set `degree = 15`, then increase `lambda` from 0 → 20 (watch validation error improve)
2. Set `lambda = 0`, increase `degree` from 1 → 20 (watch gap grow)
3. Find the sweet spot where validation MSE is minimized
""")

st.info("💡 **Key insight**: The gap (Val MSE - Train MSE) shows how much your model overfits. Larger gap = more overfitting.")


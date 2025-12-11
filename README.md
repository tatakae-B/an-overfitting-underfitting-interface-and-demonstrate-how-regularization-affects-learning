# Overfitting vs Underfitting Demo

Streamlit web app to visualize how model complexity and L2 regularization
affect bias/variance on a simple synthetic regression task.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (typically http://localhost:8501).

Use the sidebar to adjust:
- Polynomial degree (complexity)
- Regularization strength (lambda)
- Dataset size, noise, and train/validation split

Watch how the train/validation errors and the fitted curve respond to the
changes to see overfitting and underfitting behaviors.


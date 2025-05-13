import pandas as pd
from sklearn.datasets import make_regression

def load_synthetic(n_samples=500, noise=10, random_state=42):
    X, y, _ = make_regression(
        n_samples=n_samples,
        n_features=3,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    df = pd.DataFrame(X, columns=["size_sqft", "bedrooms", "age_years"])
    df["price_k"] = y
    return df

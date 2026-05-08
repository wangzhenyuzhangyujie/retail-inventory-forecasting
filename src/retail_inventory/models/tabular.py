import numpy as np
import pandas as pd


def _lightgbm_available():
    try:
        import lightgbm  # noqa

        return True
    except Exception:
        return False


class QuantileGBDT:
    def __init__(self, alphas, params=None, seed=2026):
        self.alphas = list(alphas)
        self.params = params or {}
        self.seed = seed
        self.models = {}
        self.backend = "lightgbm" if _lightgbm_available() else "sklearn_gbrt"

    def fit(self, X, y):
        if self.backend == "lightgbm":
            import lightgbm as lgb

            for a in self.alphas:
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=float(a),
                    random_state=self.seed,
                    n_estimators=self.params.get("n_estimators", 450),
                    learning_rate=self.params.get("learning_rate", 0.035),
                    num_leaves=self.params.get("num_leaves", 64),
                    min_child_samples=self.params.get("min_child_samples", 40),
                    subsample=0.9,
                    colsample_bytree=0.9,
                )
                model.fit(X, y)
                self.models[a] = model
        else:
            from sklearn.ensemble import GradientBoostingRegressor

            for a in self.alphas:
                model = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=float(a),
                    n_estimators=self.params.get("n_estimators", 240),
                    learning_rate=self.params.get("learning_rate", 0.05),
                    max_depth=self.params.get("max_depth", 5),
                    min_samples_leaf=self.params.get("min_samples_leaf", 20),
                    random_state=self.seed,
                )
                model.fit(X, y)
                self.models[a] = model
        return self

    def predict(self, X):
        preds = {}
        arr = []
        for a in self.alphas:
            p = np.maximum(0.0, self.models[a].predict(X))
            preds["q_%s" % a] = p
            arr.append(p)
        stacked = np.vstack(arr).T
        stacked = np.sort(stacked, axis=1)
        for j, a in enumerate(self.alphas):
            preds["q_%s" % a] = stacked[:, j]
        return pd.DataFrame(preds)


class PointGBDT:
    def __init__(self, params=None, seed=2026):
        self.params = params or {}
        self.seed = seed
        self.backend = "lightgbm" if _lightgbm_available() else "sklearn_gbrt"
        self.model = None

    def fit(self, X, y):
        if self.backend == "lightgbm":
            import lightgbm as lgb

            self.model = lgb.LGBMRegressor(
                objective="regression",
                random_state=self.seed,
                n_estimators=self.params.get("n_estimators", 450),
                learning_rate=self.params.get("learning_rate", 0.035),
                num_leaves=self.params.get("num_leaves", 64),
                min_child_samples=self.params.get("min_child_samples", 40),
                subsample=0.9,
                colsample_bytree=0.9,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                loss="ls",
                n_estimators=self.params.get("n_estimators", 240),
                learning_rate=self.params.get("learning_rate", 0.05),
                max_depth=self.params.get("max_depth", 5),
                min_samples_leaf=self.params.get("min_samples_leaf", 20),
                random_state=self.seed,
            )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(0.0, self.model.predict(X))

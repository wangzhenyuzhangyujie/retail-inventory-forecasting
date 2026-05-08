import numpy as np
import pandas as pd
import torch
from torch import nn


def interpolate_quantiles_np(q_values, alphas, tau):
    q_values = np.asarray(q_values, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    tau = np.asarray(tau, dtype=float)
    out = np.zeros(len(q_values), dtype=float)
    for i in range(len(q_values)):
        out[i] = np.interp(tau[i], alphas, q_values[i])
    return np.maximum(0.0, out)


class _TauNet(nn.Module):
    def __init__(self, n_features):
        super(_TauNet, self).__init__()
        hidden = max(8, min(64, n_features * 2))
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CostAwareCalibrator:
    def __init__(
        self,
        alphas,
        tau_base,
        tau_floor=0.90,
        tau_eps=0.08,
        holding_cost=1.0,
        stockout_cost=5.0,
        epochs=250,
        batch_size=512,
        lr=0.01,
        weight_decay=1e-4,
        tau_regularization=0.05,
        seed=2026,
    ):
        self.alphas = np.asarray(alphas, dtype=np.float32)
        self.tau_base = float(tau_base)
        self.tau_floor = float(tau_floor)
        self.tau_eps = float(tau_eps)
        self.holding_cost = float(holding_cost)
        self.stockout_cost = float(stockout_cost)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.tau_regularization = float(tau_regularization)
        self.seed = seed
        self.model = None
        self.mean_ = None
        self.std_ = None

    def _scale(self, X, fit=False):
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0) + 1e-6
        return (X - self.mean_) / self.std_

    def _interp_torch(self, q, tau):
        alpha = torch.tensor(self.alphas, dtype=q.dtype, device=q.device)
        idx = (tau[:, None] >= alpha[None, :]).sum(dim=1) - 1
        idx = torch.clamp(idx, 0, len(self.alphas) - 2)
        a0 = alpha[idx]
        a1 = alpha[idx + 1]
        q0 = q.gather(1, idx[:, None]).squeeze(1)
        q1 = q.gather(1, (idx + 1)[:, None]).squeeze(1)
        w = (tau - a0) / torch.clamp(a1 - a0, min=1e-6)
        return q0 + w * (q1 - q0)

    def fit(self, X, q_values, y_pp):
        torch.manual_seed(self.seed)
        Xs = self._scale(X, fit=True)
        q_values = np.sort(np.asarray(q_values, dtype=np.float32), axis=1)
        y_pp = np.asarray(y_pp, dtype=np.float32)
        self.model = _TauNet(Xs.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        X_t = torch.tensor(Xs, dtype=torch.float32)
        q_t = torch.tensor(q_values, dtype=torch.float32)
        y_t = torch.tensor(y_pp, dtype=torch.float32)
        n = len(Xs)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_t[idx]
                qb = q_t[idx]
                yb = y_t[idx]
                raw = self.model(xb)
                tau = self.tau_base + self.tau_eps * torch.tanh(raw)
                tau = torch.clamp(tau, min=self.tau_floor, max=float(self.alphas[-1]))
                s = self._interp_torch(qb, tau)
                hold = torch.relu(s - yb) * self.holding_cost
                stock = torch.relu(yb - s) * self.stockout_cost
                reg = self.tau_regularization * (tau - self.tau_base).pow(2)
                loss = (hold + stock + reg).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_tau(self, X):
        Xs = self._scale(X, fit=False)
        with torch.no_grad():
            raw = self.model(torch.tensor(Xs, dtype=torch.float32)).numpy()
        tau = self.tau_base + self.tau_eps * np.tanh(raw)
        return np.clip(tau, self.tau_floor, self.alphas[-1])

    def predict_order_up_to(self, X, q_values):
        tau = self.predict_tau(X)
        s = interpolate_quantiles_np(np.sort(np.asarray(q_values, dtype=float), axis=1), self.alphas, tau)
        return s, tau

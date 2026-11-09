"""LSTM baseline — Manuscript Section 4.1.

Two-layer LSTM, 128 hidden units, dropout 0.3, Adam lr=0.001,
batch size 64, 50 training epochs. NVIDIA Tesla V100 in the paper;
here it runs on CPU unless CUDA is available.

Requires torch. If torch is not installed, icu_train.py skips
this baseline with a clear message — it does not fake a result.
Each 25-dim window is treated as a length-25 sequence with
1 feature per step (the paper does not specify a different
sequencing).
"""
import numpy as np


def train_lstm(X_train, y_train, X_test, y_test, seed=42, epochs=50):
    import torch
    import torch.nn as nn
    from evaluation.metrics import classification_metrics
    torch.manual_seed(seed)
    np.random.seed(seed)

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2,
                                batch_first=True, dropout=0.3)
            self.fc = nn.Linear(128, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return torch.sigmoid(self.fc(h[-1])).squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    Xtr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    ytr = torch.tensor(y_train, dtype=torch.float32).to(device)
    Xte = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
    model.train()
    for _ in range(int(epochs)):
        for i in range(0, len(Xtr), 64):
            opt.zero_grad()
            out = model(Xtr[i:i + 64])
            loss = loss_fn(out, ytr[i:i + 64])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        proba = model(Xte).cpu().numpy()
    pred = (proba > 0.5).astype(int)
    return classification_metrics(y_test, pred, proba)

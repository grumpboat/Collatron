import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import itertools
import os

def collatz_steps(n):
    steps = 0
    while n != 1:
        n = 3 * n + 1 if n % 2 else n // 2
        steps += 1
    return steps

def dynamic_sample(start, end, count):
    if end <= start:
        return []
    pool = list(range(start, end))
    return random.sample(pool, min(len(pool), count))

def count_trailing_zeros(n):
    if n == 0:
        return 0
    return (n & -n).bit_length() - 1

def encode_number(n):
    steps = collatz_steps(n)
    max_reached = n
    tmp = n
    for _ in range(1000):
        if tmp == 1:
            break
        tmp = 3 * tmp + 1 if tmp % 2 else tmp // 2
        max_reached = max(max_reached, tmp)

    features = [
        n / (2**20),
        math.log2(n + 1),
        bin(n).count('1') / 64,
        n % 2,
        n % 3 / 3,
        n % 5 / 5,
        n % 7 / 7,
        n % 11 / 11,
        count_trailing_zeros(n) / 20,
        sum(int(d) for d in str(n)) / 100,
        math.log2(max_reached + 1) / 20,
        max_reached / (n + 1),
        steps / 500
    ]

    tmp = n
    for _ in range(3):
        tmp = 3 * tmp + 1 if tmp % 2 else tmp // 2
        features.append(math.log2(tmp + 1) / 20)

    return torch.tensor(features, dtype=torch.float32)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = out + residual
        out = self.norm(out)
        return out

class Collatron(nn.Module):
    def __init__(self, dim=256, n_heads=4, n_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(16, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim * 4, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, 1)

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x).unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.output_proj(x)

def load_or_init_model():
    model = Collatron()
    if os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt"))
        print("Loaded existing model")
    else:
        print("No model found, starting fresh")
    return model

def predict_steps(model, n):
    model.eval()
    with torch.no_grad():
        x = encode_number(n).unsqueeze(0)
        log_pred = model(x).item()
        log_pred = torch.tensor(log_pred).clamp(0, 10).item()
        steps_pred = (2 ** log_pred) - 1
    return steps_pred

def test_range(model, samples):
    model.eval()
    errors = []
    max_error = 0
    max_err_n = None
    print("n    | Predicted | Actual | Error")
    print("-" * 40)
    with torch.no_grad():
        for n in samples:
            pred = predict_steps(model, n)
            actual = collatz_steps(n)
            error = abs(pred - actual)
            errors.append((n, error))
            if error > max_error:
                max_error = error
                max_err_n = n
            print(f"{n:<4} | {pred:9.2f} | {actual:<6} | {error:.2f}")
    avg_error = sum(e for _, e in errors) / len(errors)
    print(f"\nAverage Absolute Error: {avg_error:.3f}")
    print(f"Worst Error: n={max_err_n}, error={max_error:.2f}")
    return avg_error, max_err_n, max_error, errors

from collections import deque

def continuous_training(
    max_samples,
    batch_size,
    regret_weight,
    lr,
    total_epochs,
    worst_sample_count
):
    model = load_or_init_model()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss(reduction='none')
    max_grad_norm = 1.0

    regret_memory = deque(maxlen=1000)
    persistent_clowns = {}

    for epoch in range(total_epochs):
        if epoch < 10:
            easy = range(1, min(100, max_samples // 5))
            mid = []
            hard = []
        elif epoch < 20:
            easy = range(1, min(100, max_samples // 4))
            mid = dynamic_sample(100, min(300, max_samples // 2), 150)
            hard = []
        else:
            easy = range(1, min(50, max_samples // 2))
            mid = dynamic_sample(50, min(200, max_samples), 100)
            hard = dynamic_sample(200, max_samples, 150)

        all_samples = list(itertools.chain(easy, mid, hard))

        dataset = []
        for n in all_samples:
            x = encode_number(n)
            y_steps = collatz_steps(n)
            y = torch.tensor([math.log2(y_steps + 1)], dtype=torch.float32)
            dataset.append((n, x, y))

        model.eval()
        with torch.no_grad():
            errors = []
            for n, x, y in dataset:
                pred = model(x.unsqueeze(0)).item()
                pred_steps = max(0, (2 ** pred) - 1)
                actual_steps = max(0, (2 ** y.item()) - 1)
                err = abs(pred_steps - actual_steps)

                errors.append((n, x, y, err))
                regret_memory.append((n, err))

        for n, err in regret_memory:
            if n not in persistent_clowns:
                persistent_clowns[n] = []
            persistent_clowns[n].append(err)
        for n in list(persistent_clowns.keys()):
            if len(persistent_clowns[n]) > 5:
                persistent_clowns[n] = persistent_clowns[n][-5:]

        persistent_bad = sorted(
            [(n, sum(v) / len(v)) for n, v in persistent_clowns.items() if len(v) >= 3],
            key=lambda x: x[1], reverse=True
        )
        persistent_sample_ids = [x[0] for x in persistent_bad[:worst_sample_count]]

        errors.sort(key=lambda e: e[3], reverse=True)
        worst_samples = [e for e in errors if e[0] in persistent_sample_ids][:worst_sample_count]
        other_samples = random.sample(errors, min(batch_size, len(errors)))
        train_batch = worst_samples + other_samples
        random.shuffle(train_batch)

        model.train()
        losses = []
        for i in range(0, len(train_batch), batch_size):
            batch = train_batch[i:i + batch_size]
            if not batch:
                continue

            xs = torch.stack([b[1] for b in batch])
            ys = torch.stack([b[2] for b in batch]).view(-1)
            preds = model(xs).view(-1).clamp(0, 10)

            pred_steps = (2 ** preds) - 1
            actual_steps = (2 ** ys) - 1
            errors = (pred_steps - actual_steps).abs()

            weights = 1.0 + regret_weight * torch.log1p(errors)
            loss = (loss_fn(preds, ys) * weights).mean()
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e5, neginf=1e5)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            losses.append(loss.item())

        scheduler.step()
        avg_loss = sum(losses) / len(losses)

        if epoch % 10 == 0 or epoch < 5 or epoch == total_epochs - 1:
            print(f"\nEPOCH {epoch + 1} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            print("Worst cases this epoch:")
            for n, _, _, err in worst_samples[:10]:
                print(f"  n={n} => error={err:.2f}")
            print("-" * 40)

        try:
            torch.save(model.state_dict(), "model.pt")
        except Exception as e:
            print(f"Error saving model: {e}")

        if avg_loss < 1e-4:
            print("Model too good, no training needed anymore")
            break

if __name__ == "__main__":

    continuous_training(
        500,
        256,
        0.05,
        1e-4,
        10000,
        50
    )

import torch
import math
from train import Collatron, encode_number

model = Collatron()
model.load_state_dict(torch.load("model.pt"))
model.eval()

while True:
    user_input = input()

    if user_input.lower() in ["exit", "quit", "q", "stop"]:
        print("Ok bye")
        break

    if not user_input.isdigit():
        print("Invalid number. Try again.")
        continue

    n = int(user_input)

    try:
        x = encode_number(n)
        with torch.no_grad():
            pred = model(x.unsqueeze(0)).item()
            pred = max(0, min(pred, 10))

        pred_steps = 2 ** pred - 1
        print(f"Estimated Collatz steps for {n}: {pred_steps} steps")
        print("-" * 50)
    except Exception as e:
        print(f"SOMETHING WENT WRONG: {e}")
        continue

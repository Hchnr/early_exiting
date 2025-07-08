import torch

from safetensors import safe_open

MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B/model.safetensors"


with safe_open(MODEL_PATH, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        if key == "model.embed_tokens.weight":
            for i in range(16):
                t = tensor[i].to(torch.float32)
                print("-" * 80)
                print(f"token[{i}]")
                print(f"mean: {t.mean(dim=0, keepdim=True).item()}")
                print(f"var: {t.var(dim=0, keepdim=True).item()}")
                print(f"q99999: {t.quantile(0.99999, dim=0, keepdim=True).item()}")
                print(f"q9999: {t.quantile(0.9999, dim=0, keepdim=True).item()}")
                print(f"q999: {t.quantile(0.999, dim=0, keepdim=True).item()}")
                print(f"q99: {t.quantile(0.99, dim=0, keepdim=True).item()}")
                print(f"q90: {t.quantile(0.9, dim=0, keepdim=True).item()}")
                print(f"q50: {t.quantile(0.5, dim=0, keepdim=True).item()}")
                print(f"q10: {t.quantile(0.1, dim=0, keepdim=True).item()}")
                print(f"q01: {t.quantile(0.01, dim=0, keepdim=True).item()}")
                print(f"q001: {t.quantile(0.001, dim=0, keepdim=True).item()}")
                print(f"q0001: {t.quantile(0.0001, dim=0, keepdim=True).item()}")
                print(f"q00001: {t.quantile(0.00001, dim=0, keepdim=True).item()}")

import torch

from safetensors import safe_open

MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B/model.safetensors"
TOPK = 2

with safe_open(MODEL_PATH, framework="pt") as f:
    lm_head = None
    for key in f.keys():
        if key == "lm_head.weight":
            lm_head = f.get_tensor(key).transpose(0,1)
            break
    for key in f.keys():
        emb = f.get_tensor(key)
        if key == "model.embed_tokens.weight":
            logits =  emb @ lm_head
            topk = logits.topk(TOPK, sorted=True)
            for i in range(topk.indices.shape[0]):
                if topk.indices[i][0] != i:
                    print("-" * 80)
                    print(f"token[{i}] topk: {topk.indices[i]}")

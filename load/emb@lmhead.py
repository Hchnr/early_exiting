import torch

from safetensors import safe_open

MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B/model.safetensors"
TOKEN_NUM = 128
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
            logits =  emb[:TOKEN_NUM] @ lm_head
            topk = logits.topk(TOPK, sorted=True)
            print("-" * 80)
            print(f"topk for emb[:{TOKEN_NUM}]: {topk}")

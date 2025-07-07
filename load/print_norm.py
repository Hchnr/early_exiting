from safetensors import safe_open

MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B/model.safetensors"


with safe_open(MODEL_PATH, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        # print(f"{key:50}: {tensor.shape} {tensor.dtype}")
        if key == "model.norm.weight":
            topk = tensor.topk(32, sorted=True)
            print("-" * 80)
            print(topk)
            print("-" * 80)

            norms = tensor.tolist()
            for i in range(128):
                print({f"{j:4}": f"{norms[j]:2.2f}" for j in range(i*8, (i+1)*8)})

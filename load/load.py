from safetensors import safe_open

MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B/model.safetensors"


with safe_open(MODEL_PATH, framework="pt") as f:
    for key in f.keys():
        # metadata = f.get_tensor_meta(key)
        # print(f"Shape: {metadata['shape']}")
        # print(f"Data Type: {metadata['dtype']}")
        tensor = f.get_tensor(key)
        '''
        print("-" * 50)
        print(f"Tensor Name: {key}")
        print(f"Type       : {type(tensor)}")
        print(f"Shape      : {tensor.shape}")
        print(f"DType      : {tensor.dtype}")
        '''

        print(f"{key:50}: {tensor.shape} {tensor.dtype}")

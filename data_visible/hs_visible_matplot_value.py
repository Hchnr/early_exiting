import torch
import matplotlib.pyplot as plt
import numpy as np

NUM_LAYERS = 19
NUM_K = 8
LAYER_CHUNK = 5

logits_dict = None
with open("./token0_hs_logits_by_layer.pt", "rb") as f:
    logits_dict = torch.load(f, map_location=torch.device('cpu'))

x_pos = []
y_value = []
for i_layer in range(NUM_LAYERS):
    hs = logits_dict[f"layer{i_layer}_hs"]
    logits_topk = hs.topk(NUM_K, dim=-1)
    x_pos.extend(logits_topk[1].squeeze().tolist())
    y_value.extend(logits_topk[0].squeeze().tolist())
    print(f"x_pos[{i_layer}]  : {x_pos}")
    print(f"y_value[{i_layer}]: {y_value}")
    

plt.figure(figsize=(10, 8))
plt.title('HS Top Value on Pos')
plt.xlabel('POS')
plt.ylabel('HS')


layers = np.arange(NUM_LAYERS)
layers = layers.repeat(repeats=NUM_K)

plt.scatter(x_pos, y_value, c=layers, cmap="coolwarm", s=20, alpha=0.6)
plt.colorbar(label='iLayer')
plt.tight_layout() 
plt.show()
        
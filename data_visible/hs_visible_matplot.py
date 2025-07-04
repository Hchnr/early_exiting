import torch
import matplotlib.pyplot as plt
import numpy as np

NUM_LAYERS = 19
NUM_K = 4
LAYER_CHUNK = 5

logits_dict = None
with open("./token0_hs_logits_by_layer.pt", "rb") as f:
    logits_dict = torch.load(f, map_location=torch.device('cpu'))

x_pos = []
y_value = []
x_pos_list = []
y_value_list = []
for i_layer in range(NUM_LAYERS):
    hs = logits_dict[f"layer{i_layer}_hs"]
    logits_topk = hs.topk(NUM_K, dim=-1)
    x_tmp = logits_topk[1].squeeze().tolist()
    y_tmp = logits_topk[0].squeeze().tolist()
    x_pos.extend(x_tmp)
    y_value.extend(y_tmp)
    x_pos_list.append(x_tmp)
    y_value_list.append(y_tmp)

for i_layer in range(NUM_LAYERS):
    x_tmp = x_pos_list[i_layer]
    print(f"x[{i_layer:2}]: {x_tmp}")

for i_layer in range(NUM_LAYERS):
    y_tmp = y_value_list[i_layer]
    print(f"y[{i_layer:2}]: {y_tmp}")

plt.figure(figsize=(10, 8))
plt.title('HS Top Value on Pos')
plt.xlabel('POS')
plt.ylabel('Layer')

layers = np.arange(NUM_LAYERS)
layers = layers.repeat(repeats=NUM_K)

plt.scatter(x_pos, layers, cmap="coolwarm", s=20, alpha=0.6)
# plt.colorbar(label='iLayer')
plt.tight_layout() 
plt.show()
        
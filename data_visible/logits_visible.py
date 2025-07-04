import torch
import plotext as plt

NEED_PLOT = False

logits_dict = None
with open("./token0_hs_logits_by_layer.pt", "rb") as f:
    logits_dict = torch.load(f)

for i_layer in range(19):
    logits = logits_dict[f"layer{i_layer}_logits"]

    logits_topk = logits.topk(8, dim=-1)
    x_pos = logits_topk[1].squeeze().tolist()
    y_value = logits_topk[0].squeeze().tolist()
    
    print(f"x_pos: {x_pos}")
    print(f"y_value: {y_value}")

    if NEED_PLOT:
        plt.plot(x_pos, y_value)
        plt.title(f"Layer-{i_layer} Logits Topk")
        plt.xlabel("pos")
        plt.ylabel("logit")
        plt.show()

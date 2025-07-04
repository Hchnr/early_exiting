import torch


NEED_PLOT = True
LIB_PLOT = "matplotlib" # matplotlib, plotext, termplotlib, asciichartpy

logits_dict = None
with open("./token0_hs_logits_by_layer.pt", "rb") as f:
    logits_dict = torch.load(f, map_location=torch.device('cpu'))

for i_layer in range(19):
    hs = logits_dict[f"layer{i_layer}_hs"]

    logits_topk = hs.topk(8, dim=-1)
    x_pos = logits_topk[1].squeeze().tolist()
    y_value = logits_topk[0].squeeze().tolist()
    
    print(f"x_pos: {x_pos}")
    print(f"y_value: {y_value}")

    if NEED_PLOT:
        if LIB_PLOT == "matplotlib":
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(8, 5))
           
            # colors = np.random.rand(50)
            # sizes = 1000 * np.random.rand(50)
            # plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
            plt.scatter(x_pos, y_value)
            plt.title('HS on POS')
            plt.xlabel('POS')
            plt.ylabel('HS')
            plt.colorbar()
            plt.show()
        elif LIB_PLOT == "plotext":
            import plotext as plt
            plt.scatter(x_pos, y_value, marker='o')
            plt.title(f"Layer-{i_layer} HiddenStates Topk")
            plt.xlabel("pos")
            plt.ylabel("HS")
            for x, y in zip(x_pos, y_value):
                plt.text(f"({x}, {y})", x, y, color='red')
            plt.show()
        elif LIB_PLOT == "termplotlib":
            import termplotlib as tpl
            fig = tpl.figure()
            fig.plot(x_pos, y_value, label="sin(x)", width=80, height=20)
            fig.show()
        elif LIB_PLOT == "asciichartpy":
            import asciichartpy as ac
            options = {
                'title': '正弦函数曲线',
                'xlabel': 'x',
                'ylabel': 'sin(x)*5+5',
                'height': 20,
                'format': '{:4.1f}',
                'padding': {'top': 1, 'bottom': 1, 'left': 4, 'right': 4},
                'grid': True,
                'max': 10,
                'min': 0
            }
            chart = ac.plot((y_value, x_pos), options)
            print(chart)

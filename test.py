from tools.confusion import xywh2xyxy
import torch

detects = [430, 250, 987, 503]
labs = [423, 240, 980, 500]

box4 = torch.from_numpy(detects)
box3 = torch.from_numpy(labs)

d = torch.tensor(xywh2xyxy(box4))
l= torch.tensor(xywh2xyxy(box3))
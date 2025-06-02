import torch
import torchvision
import torch.nn as nn

print(">>> 开始构造模型")
model = torchvision.models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 8)

print(">>> 加载 state_dict")
state_dict = torch.load("model/resnet50_model.pth", map_location='cpu')  # 确保路径正确
print(">>> 权重成功加载")

model.load_state_dict(state_dict)
print(">>> 模型结构填充完毕")

model.eval()
print(">>> 测试完成 ✅")
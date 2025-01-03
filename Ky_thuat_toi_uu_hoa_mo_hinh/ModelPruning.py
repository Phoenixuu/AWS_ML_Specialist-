import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils import prune

# Định nghĩa mô hình
model = torchvision.models.resnet18(pretrained=True)

# Prune 20% các trọng số trong tất cả các lớp của mô hình
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.layer1[0].conv1, 'weight'),
    (model.layer1[0].conv2, 'weight'),
    (model.layer2[0].conv1, 'weight'),
    (model.layer2[0].conv2, 'weight'),
    (model.layer3[0].conv1, 'weight'),
    (model.layer3[0].conv2, 'weight'),
    (model.layer4[0].conv1, 'weight'),
    (model.layer4[0].conv2, 'weight'),
)

# Pruning 20% trọng số ở mỗi lớp
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

# Kiểm tra số lượng trọng số sau khi pruning
print("Number of pruned weights in conv1:", torch.sum(model.conv1.weight == 0).item())
print("Number of pruned weights in layer1[0].conv1:", torch.sum(model.layer1[0].conv1.weight == 0).item())

# Đào tạo mô hình (giống như đào tạo thông thường)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
model.train()
for epoch in range(5):
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Định nghĩa mô hình Teacher (mô hình lớn)
teacher_model = torchvision.models.resnet18(pretrained=True)
teacher_model.eval()  # Đặt chế độ evaluation cho Teacher model

# Định nghĩa mô hình Student (mô hình nhỏ hơn)
student_model = torchvision.models.resnet18(pretrained=False)
student_model.fc = nn.Linear(student_model.fc.in_features, 10)  # Cải tiến đầu ra để phù hợp với 10 lớp

# Loss function cho distillation (sử dụng cả CrossEntropy và KL Divergence)
def distillation_loss(y, labels, teacher_scores, T, alpha):
    """
    Compute distillation loss between teacher and student.
    """
    # Loss giữa dự đoán của Student và Teacher
    hard_loss = nn.CrossEntropyLoss()(y, labels)
    soft_loss = nn.KLDivLoss()(nn.functional.log_softmax(y/T, dim=1), nn.functional.softmax(teacher_scores/T, dim=1))
    return alpha * hard_loss + (1 - alpha) * soft_loss * (T * T)

# Training
def train_distillation(student_model, teacher_model, trainloader, optimizer, epochs, T=3, alpha=0.7):
    for epoch in range(epochs):
        student_model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()

            # Forward pass qua mô hình teacher và student
            teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)
            
            # Tính toán loss distillation
            loss = distillation_loss(student_outputs, labels, teacher_outputs, T, alpha)
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Dataloader cho dataset CIFAR10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Optimizer cho student model
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# Huấn luyện mô hình Student với distillation
train_distillation(student_model, teacher_model, trainloader, optimizer, epochs=5)


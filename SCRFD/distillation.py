import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, student_output, teacher_output, labels):
        """Calculate distillation loss (combination of soft targets and hard targets)."""
        # Soft loss (from teacher)
        soft_loss = F.kl_div(F.log_softmax(student_output / self.temperature, dim=1),
                             F.softmax(teacher_output / self.temperature, dim=1),
                             reduction='batchmean') * (self.temperature ** 2)
        
        # Hard loss (from true labels)
        hard_loss = F.cross_entropy(student_output, labels)
        
        # Total loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def train_student(self, train_loader, epochs=5, learning_rate=1e-4):
        """Train the student model using knowledge distillation."""
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.student_model.train()
            for data, target in train_loader:
                optimizer.zero_grad()

                # Get outputs from teacher and student models
                teacher_output = self.teacher_model(data)
                student_output = self.student_model(data)

                # Calculate loss
                loss = self.distillation_loss(student_output, teacher_output, target)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


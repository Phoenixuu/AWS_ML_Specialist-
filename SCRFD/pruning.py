import torch
import torch.nn as nn
import torch.nn.functional as F

class Pruner:
    def __init__(self, model, pruning_rate=0.2):
        self.model = model
        self.pruning_rate = pruning_rate

    def prune(self):
        """Prune the model by removing weights with the smallest magnitude."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    # Calculate the pruning threshold
                    threshold = torch.quantile(param.abs(), self.pruning_rate)
                    mask = param.abs() > threshold
                    param.data.mul_(mask)

    def fine_tune(self, epochs=5, learning_rate=1e-4):
        """Fine-tune the pruned model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Fine-tuning loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(inputs)  # inputs should be your data here
            loss = criterion(output, labels)  # labels should be your target here
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


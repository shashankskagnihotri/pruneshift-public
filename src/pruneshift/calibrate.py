import torch
import torch.optim as optim
import torch.nn as nn


class Calibration(nn.Module):
    def __init__(self, network):
        super(Calibration, self).__init__()
        self.network=network
        self.temperature = nn.Parameter(torch.ones(1))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
        self.logits = torch.ones(1)
        self.labels = torch.ones(1)

    def forward(self, images, labels):
        self.network.eval()
        with torch.no_grad():
            logits = self.network(images)
        self.logits = logits
        self.labels = labels
        def _eval():
            loss = self.criterion(torch.div(self.logits, self.temperature), self.labels)
            loss.backward()
            return loss

        self.optimizer.step(_eval)

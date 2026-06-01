
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Expert class
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.fc(x)

# Define the Gating Network class
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=2)

# Define the Mixture of Experts Layer class
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x, num_experts_per_tok):
        gating_scores = self.gate(x)
        topk_gating_scores, topk_indices = gating_scores.topk(num_experts_per_tok, dim=2, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=2)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)
        return output
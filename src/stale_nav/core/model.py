"""Dimension-explicit actor and critic; no platform or feature assumptions."""
import math
import torch
from torch import nn
from torch.distributions import Categorical

from .ports import EnvironmentSpec


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_count: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.spec = EnvironmentSpec(observation_size, action_count)
        if any(type(h) is not int or h <= 0 for h in hidden_sizes):
            raise ValueError("Hidden dimensions must be positive integers")

        def network(output: int, gain: float):
            layers = []
            previous = observation_size
            for hidden in hidden_sizes:
                layer = nn.Linear(previous, hidden)
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
                layers.extend((layer, nn.Tanh()))
                previous = hidden
            last = nn.Linear(previous, output)
            nn.init.orthogonal_(last.weight, gain)
            nn.init.zeros_(last.bias)
            return nn.Sequential(*layers, last)

        self.actor = network(action_count, 0.01)
        self.critic = network(1, 1.0)

    def forward(self, features: torch.Tensor):
        return self.actor(features), self.critic(features).squeeze(-1)

    def sample(self, features: torch.Tensor):
        logits, values = self(features)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values

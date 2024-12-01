import torch
import random

class Q_net(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super(Q_net, self).__init__()

        self.action_space = action_space

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_space, 256),            
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_space)
        )
    def forward(self, x):        
        y = self.layers(x)
        return y    

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space-1)
        else:
            y = self.forward(obs)
            return y.argmax().item()


class Q_net512(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super(Q_net512, self).__init__()

        self.action_space = action_space

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_space, 512),            
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_space)
        )
    def forward(self, x):        
        y = self.layers(x)
        return y    

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space-1)
        else:
            y = self.forward(obs)
            return y.argmax().item()

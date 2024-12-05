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



class Greedy_Q_net(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super(Greedy_Q_net, self).__init__()

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
        if True:
            heatmap_score = obs[[39, 31, 41, 39]]            
            return heatmap_score.argmax().item()
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
        


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, gamma=0.99):
        super(Policy, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_space, 256),                    
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),            
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),            
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_space),            
            torch.nn.Softmax()
        )


        # Episode policy and reward history 
        self.policy_history = torch.autograd.Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

        self.gamma = 0.99


    def forward(self, x):        
        y = self.layers(x)

        return y
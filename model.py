import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import os

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions,lr, model_dir, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_dir = model_dir
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.ln1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.ln2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        nn.init.uniform_(self.mu.weight.data, -f3, f3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        mu = torch.tanh(self.mu(x))
        return mu

    def save_checkpoint(self, model_file):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), os.path.join(self.model_dir, model_file))

    def load_checkpoint(self, model_file):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(os.path.join(self.model_dir, model_file)))

class CriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, lr, model_dir, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_dir = model_dir
        self.name = name

        # Define and randomly initialize the weights of the network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        self.ln1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.fca = nn.Linear(self.n_actions, self.fc2_dims)
        q = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -q, q)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.ln1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.ln2(state_value)

        action_value = self.fca(action)    

        state_action_value = F.relu(torch.add(state_value, action_value))

        q = self.q(state_action_value)
        return q

    def save_checkpoint(self, model_file):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), os.path.join(self.model_dir, model_file))

    def load_checkpoint(self, model_file):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(os.path.join(self.model_dir, model_file)))

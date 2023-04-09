import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer
from noise import OUActionNoise
from her import HindsightExperienceReplay

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=3,
                    max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64, args=None, warmup=1000):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.distance_threshold = 0.05
        self.env = env
        self.her = HindsightExperienceReplay('future')
        self.warmup = warmup
        self.time_step = 0

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, lr=args.lr_actor, model_dir='models', name='actor')
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, lr=args.lr_critic, model_dir='models', name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, lr=args.lr_actor, model_dir='models', name='target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, lr=args.lr_critic, model_dir='models', name='target_critic')

        # Implement Noise
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            self.time_step += 1
            return torch.tensor(np.random.normal(scale=0.1, size=(self.n_actions,)), 
                                device=self.actor.device).cpu().detach().numpy()
        else:
            self.time_step += 1
            self.actor.eval()
            observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
            mu = self.actor.forward(observation)
            mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
            mu_prime = torch.clamp(mu_prime, self.env.action_space.low[0], self.env.action_space.high[0])
            self.actor.train()
            return mu_prime.cpu().detach().numpy()
        

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     # This is the reward function used in the environment's source code, for sparse reward
    #     assert achieved_goal.shape == desired_goal.shape
    #     d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    #     return - np.array(d > self.distance_threshold, dtype=np.float64)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def play_episode(self, g, transition):
        observation, action, reward, new_observation, done, info = transition
        
        new_observation['observation'] = new_observation['observation'].reshape(1, -1)
        new_observation['desired_goal'] = new_observation['desired_goal'].reshape(1, -1)
        new_observation['achieved_goal'] = new_observation['achieved_goal'].reshape(1, -1)
        new_observation = np.concatenate([new_observation['observation'], new_observation['desired_goal']], axis=1)
        obs = observation[:,:6]
        new_obs = new_observation[:,:6]
        g = g.reshape(1, -1)

        reward_ = self.env.compute_reward(obs[:,:3], g, None)
        
        observation = np.concatenate([obs, g], axis=1)
        new_observation = np.concatenate([new_obs, g], axis=1)

        self.remember(observation, transition[1], reward_, new_observation, transition[4])

    def learn(self, minibatch):
        state, action, reward, new_state, done = minibatch
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done, dtype=torch.float).to(self.actor.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        target_critic_value = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*target_critic_value[j]*done[j])

        target = torch.tensor(target).to(self.actor.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_params = dict(self.critic.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())

        for name in critic_params:
            critic_params[name] = tau*critic_params[name].clone() + \
                                    (1-tau)*target_critic_params[name].clone()

        self.target_critic.load_state_dict(critic_params, strict=False)

        for name in actor_params:
            actor_params[name] = tau*actor_params[name].clone() + \
                                    (1-tau)*target_actor_params[name].clone()

        self.target_actor.load_state_dict(actor_params, strict=False)

    def save_models(self, episode_no):
        print('... saving models ...')
        self.actor.save_checkpoint('%s_%d' % (self.actor.name, episode_no))
        self.critic.save_checkpoint('%s_%d' % (self.critic.name, episode_no))
        self.target_actor.save_checkpoint('%s_%d' % (self.target_actor.name, episode_no))
        self.target_critic.save_checkpoint('%s_%d' % (self.target_critic.name,episode_no))

    def load_models(self, episode_no):
        print('... loading models ...')
        self.actor.load_checkpoint('%s_%d' % (self.actor.name, episode_no))
        self.critic.load_checkpoint('%s_%d' % (self.critic.name, episode_no))
        self.target_actor.load_checkpoint('%s_%d' % (self.target_actor.name, episode_no))
        self.target_critic.load_checkpoint('%s_%d' % (self.target_critic.name,episode_no))

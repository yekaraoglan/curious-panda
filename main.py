# TODO: Import necessary modules
import gym
import panda_gym
import numpy as np
import argparse
from agent import Agent

# TODO: Define a main function
def main(args, env, agent):
    # train
    for epoch in range(args.num_epochs): # 200 epochs
        print("Epoch: {}".format(epoch))
        for cycle in range(args.num_cycles): # 50 cycles
            print("Cycle: {}".format(cycle))
            for episode in range(args.num_episodes): # 16 episodes
                observation = env.reset() # initial state
                done = False
                episode_transitions = []
                score = 0
                while not done:
                    observation['observation'] = observation['observation'].reshape(1, -1)
                    observation['desired_goal'] = observation['desired_goal'].reshape(1, -1)
                    observation['achieved_goal'] = observation['achieved_goal'].reshape(1, -1)
                    observation = np.concatenate([observation['observation'], observation['desired_goal']], axis=1)

                    action = agent.choose_action(observation).flatten()
                    next_obs, reward, done, info = env.step(action)
                    transition = (observation, action, reward, next_obs, done, info)
                    episode_transitions.append(transition)
                    score += reward
                    observation = next_obs

                desired_goal = observation['desired_goal']

                for i,transition in enumerate(episode_transitions):
                    agent.play_episode(desired_goal, transition)
                    
                    goals = agent.her.sample_goals(i, episode_transitions, 4)
                    # Sample batch of goals
                    for g in goals:
                        achieved_goal = g[:,:3]
                        agent.play_episode(achieved_goal, transition)

                for _ in range(args.num_updates): # 40 updates
                    # Sample a minibatch of transitions
                    minibatch = agent.memory.sample_buffer(args.batch_size)
                    # Perform optimization step
                    agent.learn(minibatch)

                print("Episode: {}, Score: {}".format(episode, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=16)
    parser.add_argument('--num_cycles', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_updates', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_actor', type=float, default=0.000025)
    parser.add_argument('--lr_critic', type=float, default=0.00025)
    parser.add_argument('--alpha', type=float, default=0.0003)
    parser.add_argument('--beta', type=float, default=0.0003)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--monitor_freq', type=int, default=200)
    parser.add_argument('--model_save_freq', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=1000)

    args = parser.parse_args()

    env = gym.make('PandaReach-v2', render=True)
    env = gym.wrappers.Monitor(env, "recordings", video_callable=lambda episode_id: episode_id%args.monitor_freq==0, force=True)

    # agent
    agent = Agent(alpha=args.alpha, beta=args.beta, input_dims=[9], tau=args.tau, env=env, gamma=0.99, n_actions=env.action_space.shape[0],
                    max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64, args=args, warmup=args.warmup)
    
    main(args, env, agent)

    
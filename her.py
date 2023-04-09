import numpy as np

class HindsightExperienceReplay:
    def __init__(self, goal_strategy):
        self.goal_strategy = goal_strategy

    def sample_goals(self, t, episode, batch_size):
        episode = np.array(episode)
        if self.goal_strategy == 'future': 
            # 'future' strategy is the best performing strategy 
            # according to the experiments in the paper
            if len(episode) < t:
                return None
            if len(episode) < t + batch_size:
                indices = np.random.choice(len(episode) - t, len(episode) - t, replace=False)
                indices += t
                return [episode[indices][k][0] for k in range(len(episode) - t)]
            else:
                indices = np.random.choice(len(episode) - t, batch_size, replace=False)
                indices += t
                return [episode[indices][k][0] for k in range(batch_size)]
        # TODO: implement other strategies
            
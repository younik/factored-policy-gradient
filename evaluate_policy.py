import numpy as np
import torch


def evaluate_policy(env, policy, num_episodes):
    rewards = np.zeros((num_episodes, env.num_envs))

    for ep_id in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs = np.asarray(obs)
            with torch.no_grad():
                actions = policy(torch.from_numpy(obs))
            obs, rew, ter, tru, _ = env.step(actions.numpy())

            assert np.all(ter == False)
            done = tru[0]
            assert np.all(done == tru)
            assert done or np.all(rew == 0)
        
        assert np.all(rew != 0)
        rewards[ep_id] = rew
    
    return np.mean(rewards)
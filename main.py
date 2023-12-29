from absl import app, flags
import breedgym
from chromax import Simulator
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from policy_net import PolicyNet
import wandb
from evaluate_policy import evaluate_policy


flags.DEFINE_float("lr", 1e-4, "Optimizer learning rate")
flags.DEFINE_integer("num_episodes", 100, "Number of episodes per epoch")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")


def train_epoch(train_env, policy, action_std, optimizer, num_episodes):
    # ------------- collect rollout -------------
    steps_per_ep = train_env.num_generations
    num_envs = train_env.num_envs
    rewards = np.zeros((num_episodes, num_envs, 1))
    log_probs = torch.zeros((num_episodes, num_envs, steps_per_ep))

    for ep_id in range(num_episodes):
        obs, _ = train_env.reset()
        for step_id in range(steps_per_ep):
            obs = np.asarray(obs)
            action_mean = policy(torch.from_numpy(obs).to(policy.device))
            matrix_std = torch.ones_like(action_mean) * action_std
            distribution = Normal(action_mean, matrix_std)
            actions = distribution.sample()
            log_probs[ep_id, :, step_id] = distribution.log_prob(actions).sum((-1, -2))
            obs, rew, ter, tru, _ = train_env.step(actions.numpy())

            assert np.all(ter == False)
            done = step_id == steps_per_ep - 1
            assert np.all(done == tru)
            assert done or np.all(rew == 0)
        
        assert np.all(rew != 0)
        rewards[ep_id] = rew[:, None]

    # ------------- optimize -------------
    assert np.all(rewards != 0)
    rew_mean = rewards.mean()
    gs = F.normalize(torch.from_numpy(rewards - rew_mean))
    loss = -torch.mean(gs * log_probs) 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return rew_mean, loss.item()


def main(_):
    config = flags.FLAGS
    wandb.init(
        "Standard",
        project="FactoredMDP",
    )

    simulator = Simulator(genetic_map="genetic_map.txt")
    initial_pop = simulator.load_population("geno.npy")[:50]
    del simulator
    train_env = gym.make(
        "PairScores",
        num_envs=8,
        initial_population=initial_pop,
        genetic_map="genetic_map.txt",
        num_generations=5,
    )

    baseline_rew = evaluate_policy(
        train_env,
        policy=lambda _ : np.random.randn(*train_env.action_space.shape),
        num_episodes=200
    )
    print("RANDOM POLICY:", baseline_rew)


    policy = PolicyNet()
    # check if you need logstd instead of std
    action_std = nn.Parameter(torch.ones(1,), requires_grad=True)
    optimizer = torch.optim.Adam(list(policy.parameters()) + [action_std], lr=config.lr)

    for i in range(config.num_epochs):
        rew_mean, loss = train_epoch(train_env, policy, action_std, optimizer, num_episodes=config.num_episodes)
        wandb.log({"mean_rew": rew_mean})
        wandb.log({"loss": loss})
        wandb.log({"action_std": action_std.item()})
        wandb.log({"epoch": i + 1})
        wandb.log({"env_episodes": (i + 1) * config.num_episodes})


if __name__ == "__main__":
    app.run(main)


    


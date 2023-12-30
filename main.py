from absl import app, flags
import breedgym
from chromax import Simulator
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from policies.policy_net import PolicyNet
import wandb
from evaluate_policy import evaluate_policy
from functools import partial


flags.DEFINE_float("lr", 5e-4, "Optimizer learning rate")
flags.DEFINE_integer("num_episodes", 100, "Number of episodes per epoch")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")
flags.DEFINE_integer("num_envs", 8, "Number of parallel environments")
flags.DEFINE_integer("eval_freq", 5, "Frequency of evaluation (in epochs)")
flags.DEFINE_integer("eval_episodes", 25, "Number of episodes for evaluation")
flags.DEFINE_integer("individual_per_gen", 25, "Individual per generation in the breeding program")
flags.DEFINE_integer("trials", 10, "Number of trials")
flags.DEFINE_bool("factorize", False, "Whether to use factorized learning algorithm or not")
flags.DEFINE_string("device", "auto", "Which device to use")


def train_epoch(train_env, policy, action_std, optimizer, num_episodes, factorize):
    # ------------- collect rollout -------------
    steps_per_ep = train_env.num_generations
    num_envs = train_env.num_envs
    ind = train_env.individual_per_gen
    rewards = np.zeros((num_episodes, num_envs))
    if factorize:
        log_probs = torch.zeros((num_episodes, num_envs, steps_per_ep, ind, ind))
        values = np.zeros((num_episodes, num_envs, ind))
        low_level_actions = np.zeros((num_episodes, num_envs, steps_per_ep, ind, 2), dtype=np.int32)
    else:
        values = np.zeros((num_episodes, num_envs, 1))
        log_probs = torch.zeros((num_episodes, num_envs, steps_per_ep))

    for ep_id in range(num_episodes):
        obs, _ = train_env.reset()
        for step_id in range(steps_per_ep):
            obs = np.asarray(obs)
            action_mean = policy(torch.from_numpy(obs))
            matrix_std = torch.ones_like(action_mean) * action_std
            distribution = Normal(action_mean, matrix_std)
            actions = distribution.sample()
            log_prob = distribution.log_prob(actions)
            obs, rew, ter, tru, infos = train_env.step(actions.cpu().numpy())
            assert np.all(ter == False)
            done = step_id == steps_per_ep - 1
            assert np.all(done == tru)
            assert done or np.all(rew == 0)
            if factorize:
                log_probs[ep_id, :, step_id] = log_prob
                low_level_actions[ep_id, :, step_id] = infos['low_level_actions']                
            else:
                log_probs[ep_id, :, step_id] = log_prob.sum((-1, -2))
        
        assert np.all(rew != 0)
        rewards[ep_id] = rew
        values[ep_id] = infos['GEBV'].squeeze(-1) if factorize else rew[:, None]

    # ------------- normalize rew -------------
    assert np.all(values != 0)
    values = values - values.mean()
    values = values / max(np.std(values), 1e-8)    

    # ------------- Compute loss -------------
    if factorize:
        # Compute "Q-value" matrix of shape (ind, ind) for every sample step
        # Average the values when a cross is performed multiple times
        vmap_axes = (0, None, None, None, None)
        vunique = jax.vmap(
            jax.vmap(
                jax.vmap(
                    partial(jnp.unique, size=ind, fill_value=0), 
                    in_axes=vmap_axes,
                ),
                in_axes=vmap_axes,
            ),
            in_axes=vmap_axes,
        )
        unique_crosses, inverse, counts = vunique(low_level_actions, False, True, True, 0)
        vbincount = jax.vmap(jax.vmap(partial(jnp.bincount, length=ind)))
        
        q_values = np.zeros((num_episodes, num_envs, steps_per_ep, ind, ind))
        current_values = values
        arange_episodes = np.arange(num_episodes)[:, None, None]
        arange_envs = np.arange(num_envs)[:, None]
        for step_id in reversed(range(steps_per_ep)):
            averaged_values = vbincount(inverse[:, :, step_id], current_values) / counts[:, :, step_id]
            averaged_values = np.nan_to_num(averaged_values)
            
            first_gen, second_gen = unique_crosses[:, :, step_id, :, 0], unique_crosses[:, :, step_id, :, 1]
            q_values[arange_episodes, arange_envs, step_id, first_gen, second_gen] = averaged_values
            
            # compute values for the previous population
            # cross [x, y] is not registered as cross [y, x], average both rows and column for next value
            sum_values = q_values[:, :, step_id].sum(-2) + q_values[:, :, step_id].sum(-1)
            count_values = np.count_nonzero(q_values[:, :, step_id], axis=-2) + np.count_nonzero(q_values[:, :, step_id], axis=-1)
            current_values = sum_values / np.maximum(count_values, 1e-8)

        loss = torch.from_numpy(q_values) * log_probs
        loss = loss.sum(dim=(3, 4))
    else:
        # In standard REINFORCE, it is simply the return (reward in this case)
        loss = torch.from_numpy(values) * log_probs

    loss = -torch.mean(loss) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return rewards.mean(), loss.item()


def run_experiment(config):
    simulator = Simulator(genetic_map="genetic_map.txt")
    initial_pop = simulator.load_population("geno.npy")[:config.individual_per_gen]
    del simulator
    env = gym.make(
        "PairScores",
        num_envs=config.num_envs,
        initial_population=initial_pop,
        genetic_map="genetic_map.txt",
    )

    policy = PolicyNet()
    # check if you need logstd instead of std
    action_std = nn.Parameter(torch.ones((1,), device="cuda"), requires_grad=True)
    optimizer = torch.optim.Adam(list(policy.parameters()) + [action_std], lr=config.lr)

    for i in range(config.num_epochs):
        rew_mean, loss = train_epoch(env, policy, action_std, optimizer, config.num_episodes, config.factorize)
        wandb.log({"mean_rew": rew_mean})
        wandb.log({"loss": loss})
        wandb.log({"action_std": action_std.item()})
        wandb.log({"epoch": i + 1})
        wandb.log({"env_episodes": (i + 1) * config.num_episodes})
        if (i + 1) % config.eval_freq == 0:
            eval_rew = evaluate_policy(env, policy, num_episodes=config.eval_episodes)
            wandb.log({"eval/mean_rew": eval_rew})

def main(_):
    config = flags.FLAGS
    if config.device == "auto":
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(config.trials):
        run = wandb.init(
            name=f"trial-{i}",
            project="FactoredMDP",
            group=f"{'Factorized' if config.factorize else 'Standard'}-{config.individual_per_gen}"
        )
        run_experiment(config)
        run.finish()


if __name__ == "__main__":
    app.run(main)


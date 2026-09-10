"""
Trains a PPO agent on the sdc-highway-v0 environment — the "agents for
autonomous driving" half of the bark-ml-inspired module (bark-ml itself
ships PPO/SAC agents via TF-Agents for its highway/merging/intersection
environments; this uses Stable-Baselines3, a more common/lighter choice
for a NumPy-only Gym env like this one).

Usage:
    python train_agent.py --timesteps 200000
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gym_env  # noqa: F401  (registers "sdc-highway-v0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--out_model", default="behavior_ppo.zip")
    parser.add_argument("--n_envs", type=int, default=4)
    args = parser.parse_args()

    vec_env = make_vec_env("sdc-highway-v0", n_envs=args.n_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out_model)
    print(f"Saved trained behavior agent to {args.out_model}")

    # Quick evaluation rollout
    eval_env = gym.make("sdc-highway-v0")
    obs, _ = eval_env.reset()
    total_reward = 0.0
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Eval episode reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()

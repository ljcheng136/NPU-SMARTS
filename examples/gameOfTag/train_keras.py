import os

# Set pythonhashseed
os.environ["PYTHONHASHSEED"] = "0"
# Silence the logs of TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
import numpy as np

np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
import random as python_random

python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
import tensorflow as tf

tf.random.set_seed(123)

# --------------------------------------------------------------------------

import signal
import sys
import warnings
import yaml


from examples.gameOfTag import env_keras as got_env
from examples.gameOfTag import agent_keras as got_agent
from examples.gameOfTag import ppo_keras as got_ppo
from examples.gameOfTag.types import AgentType, Mode
from pathlib import Path


def main(config):

    print("[INFO] Train")
    save_interval = config["model_para"].get("save_interval", 20)
    mode = Mode(config["model_para"]["mode"])  # Mode: Evaluation or Testing

    # Traning parameters
    n_steps = config["model_para"]["n_steps"]
    max_traj = config["model_para"]["max_traj"]

    # Create env
    print("[INFO] Creating environments")
    env = got_env.TagEnvKeras(config)

    # Create agent
    print("[INFO] Creating agents")
    all_agents = {name: got_agent.TagAgentKeras(name, config) for name in env.agent_ids}
    all_predator_ids = env.predators
    all_prey_ids = env.preys

    # Create model
    print("[INFO] Creating model")
    ppo_predator = got_ppo.PPOKeras(
        AgentType.PREDATOR.value,
        config,
        all_predator_ids,
        config["env_para"]["seed"] + 1,
    )
    ppo_prey = got_ppo.PPOKeras(
        AgentType.PREY.value, config, all_prey_ids, config["env_para"]["seed"] + 2
    )

    def interrupt(*args):
        nonlocal mode
        if mode == Mode.TRAIN:
            ppo_predator.save(-1)
            ppo_prey.save(-1)
        env.close()
        print("Interrupt key detected.")
        sys.exit(0)

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, interrupt)

    print("[INFO] Batch loop")
    obs_t = env.reset()
    episode = 0
    steps_t = 0
    episode_reward_predator = 0
    episode_reward_prey = 0
    for traj_num in range(max_traj):
        [agent.reset() for _, agent in all_agents.items()]
        active_agents = {}

        print(f"[INFO] New batch data collection {traj_num}/{max_traj}")
        for cur_step in range(n_steps):

            # Update all agents which were active in this batch
            active_agents.update({agent_id: True for agent_id, _ in obs_t.items()})

            # Given state, predict action and value
            logit_t = {}
            action_t = {}
            value_t = {}
            logprobability_t = {}

            logit, action = ppo_predator.actor(obs=obs_t, train=mode)
            value = ppo_predator.critic(obs_t)
            logit_t.update(logit)
            action_t.update(action)
            value_t.update(value)

            logit, action = ppo_prey.actor(obs=obs_t, train=mode)
            value = ppo_prey.critic(obs_t)
            logit_t.update(logit)
            action_t.update(action)
            value_t.update(value)

            for agent_id, logit in logit_t.items():
                logprobability_t[agent_id] = got_ppo.logprobabilities(
                    logit, [action_t[agent_id]]
                ).numpy()[0]

            # Sample action from a distribution
            next_obs_t, reward_t, done_t, _ = env.step(action_t)
            steps_t += 1

            # Store observation, action, and reward
            for agent_id, _ in obs_t.items():
                all_agents[agent_id].add_transition(
                    observation=obs_t[agent_id],
                    action=action_t[agent_id],
                    reward=reward_t[agent_id],
                    value=value_t[agent_id],
                    logprobability=logprobability_t[agent_id],
                    done=int(done_t[agent_id]),
                )
                if AgentType.PREDATOR in agent_id:
                    episode_reward_predator += reward_t[agent_id]
                else:
                    episode_reward_prey += reward_t[agent_id]
                if done_t[agent_id] == 1:
                    # Remove done agents
                    del next_obs_t[agent_id]
                    # Print done agents
                    print(
                        f"   Done: {agent_id}. Cur_Step: {cur_step}. Step: {steps_t}."
                    )

            # Reset when episode completes
            if done_t["__all__"]:
                # Next episode
                next_obs_t = env.reset()
                episode += 1

                # Log rewards
                print(
                    f"   Episode: {episode}. Cur_Step: {cur_step}. "
                    f"Episode reward predator: {episode_reward_predator}, "
                    f"Episode reward prey: {episode_reward_prey}."
                )
                ppo_predator.write_to_tb(
                    [("episode_reward_predator", episode_reward_predator, episode)]
                )
                ppo_prey.write_to_tb(
                    [("episode_reward_prey", episode_reward_prey, episode)]
                )

                # Reset counters
                episode_reward_predator = 0
                episode_reward_prey = 0
                steps_t = 0

            # Assign next_obs to obs
            obs_t = next_obs_t

        # Skip the remainder if evaluating
        if mode == Mode.EVALUATE:
            continue

        # Compute and store last state value
        for agent_id in active_agents.keys():
            if done_t.get(agent_id, None) == 0:  # Agent not done yet
                if AgentType.PREDATOR in agent_id:
                    next_value_t = ppo_predator.critic({agent_id: next_obs_t[agent_id]})
                elif AgentType.PREY in agent_id:
                    next_value_t = ppo_prey.critic({agent_id: next_obs_t[agent_id]})
                else:
                    raise Exception(f"Unknown {agent_id}.")
                all_agents[agent_id].add_last_transition(value=next_value_t[agent_id])
            else:  # Agent is done
                all_agents[agent_id].add_last_transition(value=np.float32(0))

            # Compute generalised advantages and return
            all_agents[agent_id].finish_trajectory()

        # Elapsed steps
        step = (traj_num + 1) * n_steps

        print("[INFO] Training")
        # Train predator and prey.
        # Run multiple gradient ascent on the samples.
        active_predators = [
            all_agents[agent_id]
            for agent_id in active_agents.keys()
            if AgentType.PREDATOR in agent_id
        ]
        active_preys = [
            all_agents[agent_id]
            for agent_id in active_agents.keys()
            if AgentType.PREY in agent_id
        ]

        for policy, agents in [
            (ppo_predator, active_predators),
            (ppo_prey, active_preys),
        ]:
            update_actor(
                policy,
                agents,
                config["model_para"]["actor_train_epochs"],
                config["model_para"]["target_kl"],
                config["model_para"]["clip_ratio"],
                config["model_para"]["grad_batch"],
            )
            update_critic(
                policy,
                agents,
                config["model_para"]["critic_train_epochs"],
                config["model_para"]["grad_batch"],
            )

        # Save model
        if traj_num % save_interval == 0:
            print("[INFO] Saving model")
            ppo_predator.save(step)
            ppo_prey.save(step)

    # Close env
    env.close()


def update_actor(policy, agents, iterations, target_kl, clip_ratio, grad_batch):
    for agent in agents:
        for _ in range(iterations):
            kl = got_ppo.train_actor(
                policy=policy, agent=agent, clip_ratio=clip_ratio, grad_batch=grad_batch
            )
            if kl > 1.5 * target_kl:
                # Early Stopping
                break


def update_critic(policy, agents, iterations, grad_batch):
    for agent in agents:
        for _ in range(iterations):
            got_ppo.train_critic(policy=policy, agent=agent, grad_batch=grad_batch)


if __name__ == "__main__":
    config_yaml = (Path(__file__).absolute().parent).joinpath("got_keras.yaml")
    with open(config_yaml, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Setup GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        warnings.warn(
            f"Not configured to use GPU or GPU not available.",
            ResourceWarning,
        )
        # raise SystemError("GPU device not found")

    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    main(config=config)
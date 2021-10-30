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


from examples.gameOfTag.env_tf import TagEnvTF
from examples.gameOfTag import agent as got_agent
from examples.gameOfTag import ppo_tf as got_ppo
from examples.gameOfTag.types import AgentType, Mode
from pathlib import Path


def main(config):

    print("[INFO] Train")
    # Save and eval interval
    save_interval = config["model_para"].get("save_interval", 50)

    # Mode: Evaluation or Testing
    mode = Mode(config["model_para"]["mode"])

    # Traning parameters
    num_train_epochs = config["model_para"]["num_train_epochs"]
    n_steps = config["model_para"]["n_steps"]
    max_traj = config["model_para"]["max_traj"]
    clip_value = config["model_para"]["clip_value"]
    critic_loss_weight = config["model_para"]["critic_loss_weight"]
    ent_discount_val = config["model_para"]["entropy_loss_weight"]
    ent_discount_rate = config["model_para"]["entropy_loss_discount_rate"]

    # Create env
    print("[INFO] Creating environments")
    seed = config["env_para"]["seed"]
    ## seed = random.randint(0, 4294967295)  # [0, 2^32 -1)
    env = TagEnvTF(config, seed)

    # Create agent
    print("[INFO] Creating agents")
    all_agents = {name: got_agent.TagAgent(name, config) for name in env.agent_ids}
    all_predators_id = env.predators
    all_preys_id = env.preys

    # Create model
    print("[INFO] Creating model")
    ppo_predator = got_ppo.PPOTF(
        AgentType.PREDATOR.value,
        config,
        config["env_para"]["seed"],
        observation_spec=env.single_observation_spec(),
        action_spec=env.single_action_spec(),
    )
    ppo_prey = got_ppo.PPOTF(
        AgentType.PREY.value,
        config,
        config["env_para"]["seed"],
        observation_spec=env.single_observation_spec,
        action_spec=env.single_action_spec,
    )
    print("After ppo_predator")

    # def interrupt(*args):
    #     nonlocal mode
    #     if mode == Mode.TRAIN:
    #         ppo_predator.save(-1)
    #         ppo_prey.save(-1)
    #     env.close()
    #     print("Interrupt key detected.")
    #     sys.exit(0)

    # # Catch keyboard interrupt and terminate signal
    # signal.signal(signal.SIGINT, interrupt)

    print("[INFO] Batch loop")
    states_t = env.reset()
    episode = 0
    steps_t = 0
    episode_reward_predator = 0
    episode_reward_prey = 0
    train = True if mode == Mode.TRAIN else False

    global_step = tf.compat.v1.train.get_or_create_global_step()

    env.close()

    return

    for traj_num in range(max_traj):
        [agent.reset() for _, agent in all_agents.items()]
        active_agents = {}

        print(f"[INFO] New batch data collection {traj_num}/{max_traj}")
        for cur_step in range(n_steps):

            # Update all agents which were active in this batch
            active_agents.update({agent_id: True for agent_id, _ in states_t.items()})

            # Predict and value action given state
            actions_t = {}
            action_samples_t = {}
            values_t = {}
            (
                actions_t_predator,
                action_samples_t_predator,
                values_t_predator,
            ) = ppo_predator.act(obs=states_t, train=train)
            actions_t_prey, action_samples_t_prey, values_t_prey = ppo_prey.act(
                obs=states_t, train=train
            )
            actions_t.update(actions_t_predator)
            actions_t.update(actions_t_prey)
            action_samples_t.update(action_samples_t_predator)
            action_samples_t.update(action_samples_t_prey)
            values_t.update(values_t_predator)
            values_t.update(values_t_prey)

            # Sample action from a distribution
            action_numpy_t = {
                vehicle: action_sample_t.numpy()
                for vehicle, action_sample_t in action_samples_t.items()
            }
            next_states_t, rewards_t, dones_t, _ = env.step(action_numpy_t)
            steps_t += 1

            # Store state, action and reward
            for agent_id, _ in states_t.items():
                all_agents[agent_id].add_trajectory(
                    action=action_samples_t[agent_id],
                    value=values_t[agent_id].numpy(),
                    state=states_t[agent_id],
                    done=int(dones_t[agent_id]),
                    prob=actions_t[agent_id],
                    reward=rewards_t[agent_id],
                )
                if "predator" in agent_id:
                    episode_reward_predator += rewards_t[agent_id]
                else:
                    episode_reward_prey += rewards_t[agent_id]
                if dones_t[agent_id] == 1:
                    if (
                        not dones_t["__all__"]
                        and config["model_para"]["downgrade_rewards"]
                    ):
                        # Downgrade #n last rewards for agents which become done early.
                        downgrade_len = config["model_para"]["downgrade_rewards"]
                        rewards_len = len(all_agents[agent_id].rewards)
                        min_len = np.minimum(downgrade_len, rewards_len)
                        all_agents[agent_id].rewards[-min_len:] = [
                            all_agents[agent_id].rewards[-1]
                        ] * min_len
                    # Remove done agents
                    del next_states_t[agent_id]
                    # Print done agents
                    print(
                        f"   Done: {agent_id}. Cur_Step: {cur_step}. Step: {steps_t}."
                    )

            # Reset when episode completes
            if dones_t["__all__"]:
                # Next episode
                next_states_t = env.reset()
                episode += 1

                # Log rewards
                print(
                    f"   Episode: {episode}. Cur_Step: {cur_step}. "
                    f"Episode reward predator: {episode_reward_predator}, "
                    f"Episode reward prey: {episode_reward_prey}."
                )
                with ppo_predator.tb.as_default():
                    tf.summary.scalar(
                        "episode_reward_predator", episode_reward_predator, episode
                    )
                with ppo_prey.tb.as_default():
                    tf.summary.scalar(
                        "episode_reward_prey", episode_reward_prey, episode
                    )

                # Reset counters
                episode_reward_predator = 0
                episode_reward_prey = 0
                steps_t = 0

            # Assign next_states to states
            states_t = next_states_t

        if mode == Mode.EVALUATE:
            continue

        # Compute and store last state value
        for agent_id in active_agents.keys():
            if dones_t.get(agent_id, None) == 0:  # Agent not done yet
                if AgentType.PREDATOR.value in agent_id:
                    _, _, next_values_t = ppo_predator.act(
                        {agent_id: next_states_t[agent_id]}, train=train
                    )
                elif AgentType.PREY.value in agent_id:
                    _, _, next_values_t = ppo_prey.act(
                        {agent_id: next_states_t[agent_id]}, train=train
                    )
                else:
                    raise Exception(f"Unknown {agent_id}.")
                all_agents[agent_id].add_last_transition(
                    value=next_values_t[agent_id].numpy()
                )
            else:  # Agent is done
                all_agents[agent_id].add_last_transition(value=np.float32(0))

        for agent_id in active_agents.keys():
            # Compute generalised advantages
            all_agents[agent_id].compute_advantages()

            actions = all_agents[agent_id].actions
            action_inds = tf.stack(
                [tf.range(0, len(actions)), tf.cast(actions, tf.int32)], axis=1
            )

            # Compute old probabilities
            probs_softmax = tf.nn.softmax(all_agents[agent_id].probs)
            all_agents[agent_id].old_probs = tf.gather_nd(probs_softmax, action_inds)

        predator_total_loss = np.zeros((num_train_epochs))
        predator_actor_loss = np.zeros((num_train_epochs))
        predator_critic_loss = np.zeros(((num_train_epochs)))
        predator_entropy_loss = np.zeros((num_train_epochs))

        prey_total_loss = np.zeros((num_train_epochs))
        prey_actor_loss = np.zeros((num_train_epochs))
        prey_critic_loss = np.zeros(((num_train_epochs)))
        prey_entropy_loss = np.zeros((num_train_epochs))

        # Elapsed steps
        step = (traj_num + 1) * n_steps

        print("[INFO] Training")
        # Train predator and prey.
        # Run multiple gradient ascent on the samples.
        for epoch in range(num_train_epochs):
            for agent_id in active_agents.keys():
                agent = all_agents[agent_id]
                if agent_id in all_predators_id:
                    loss_tuple = got_ppo.train_model(
                        model=ppo_predator.model,
                        optimizer=ppo_predator.optimizer,
                        actions=agent.actions,
                        old_probs=agent.old_probs,
                        states=agent.states,
                        advantages=agent.advantages,
                        discounted_rewards=agent.discounted_rewards,
                        ent_discount_val=ent_discount_val,
                        clip_value=clip_value,
                        critic_loss_weight=critic_loss_weight,
                        grad_batch=config["model_para"]["grad_batch"],
                    )
                    predator_total_loss[epoch] += loss_tuple[0]
                    predator_actor_loss[epoch] += loss_tuple[1]
                    predator_critic_loss[epoch] += loss_tuple[2]
                    predator_entropy_loss[epoch] += loss_tuple[3]

                if agent_id in all_preys_id:
                    loss_tuple = got_ppo.train_model(
                        model=ppo_prey.model,
                        optimizer=ppo_prey.optimizer,
                        actions=agent.actions,
                        old_probs=agent.old_probs,
                        states=agent.states,
                        advantages=agent.advantages,
                        discounted_rewards=agent.discounted_rewards,
                        ent_discount_val=ent_discount_val,
                        clip_value=clip_value,
                        critic_loss_weight=critic_loss_weight,
                        grad_batch=config["model_para"]["grad_batch"],
                    )
                    prey_total_loss[epoch] += loss_tuple[0]
                    prey_actor_loss[epoch] += loss_tuple[1]
                    prey_critic_loss[epoch] += loss_tuple[2]
                    prey_entropy_loss[epoch] += loss_tuple[3]

        ent_discount_val *= ent_discount_rate

        print("[INFO] Record metrics")
        # Record predator performance
        records = []
        records.append(("predator_tot_loss", np.mean(predator_total_loss), step))
        records.append(("predator_critic_loss", np.mean(predator_critic_loss), step))
        records.append(("predator_actor_loss", np.mean(predator_actor_loss), step))
        records.append(("predator_entropy_loss", np.mean(predator_entropy_loss), step))
        ppo_predator.write_to_tb(records)

        # Record prey perfromance
        records = []
        records.append(("prey_tot_loss", np.mean(prey_total_loss), step))
        records.append(("prey_critic_loss", np.mean(prey_critic_loss), step))
        records.append(("prey_actor_loss", np.mean(prey_actor_loss), step))
        records.append(("prey_entropy_loss", np.mean(prey_entropy_loss), step))
        ppo_prey.write_to_tb(records)

        # Save model
        if traj_num % save_interval == 0:
            print("[INFO] Saving model")
            ppo_predator.save(step)
            ppo_prey.save(step)

    # Close env
    env.close()


if __name__ == "__main__":
    config_yaml = (Path(__file__).absolute().parent).joinpath("got.yaml")
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
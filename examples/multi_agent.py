import gym
from argument_parser import default_argument_parser

from smarts import sstudio
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from smarts.core.utils.episodes import episodes
from smarts.zoo.agent_spec import AgentSpec

N_AGENTS = 1
AGENT_IDS = ["Agent %i" % i for i in range(N_AGENTS)]


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.Laner, 
                max_episode_steps=max_episode_steps,
                neighborhood_vehicles = True,
                done_criteria = DoneCriteria(
                    collision = False,
                    off_road = True,
                    off_route = True,
                    on_shoulder = False,
                    wrong_way = True,
                    not_moving = False,
                    agents_alive = None,
                ),
            ),
            agent_builder=KeepLaneAgent,
        )
        for agent_id in AGENT_IDS
    }

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs=agent_specs,
        headless=False,
        sumo_headless=True,
    )

    for episode in episodes(n=50):
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            observations, rewards, dones, infos = env.step(actions)
            episode.record_step(observations, rewards, dones, infos)

            import time
            time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("multi-agent-example")
    args = parser.parse_args()

    args.scenarios = [
        "/home/adai/workspace/competition_bundle/eval_scenarios/intersection/1_to_3lane_left_turn_c_rotated_agents_1"
        ]

    sstudio.build_scenario(scenario=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )

"""
Instructions to visualize
Change the paths appropriately

1) git checkout visualize
2) Open a separate new terminal and run 
    $ cd <path>/SMARTS
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip
    $ pip install -e .[camera-obs]
    $ scl envision start -s /home/adai/workspace/competition_bundle/eval_scenarios/
3) Change scenario path to the desired scenario in `args.scenario` in `__main__` of this file.
4) Change N_AGENTS at line 10 of this file to the number of agents present in the desired scenario.
5) Open a separate new terminal and run 
    $ cd <path>/SMARTS
    $ source ./.venv/bin/activate
    $ python3.8 examples/multi_agent.py
6) The simulation has been purposely slowed down by adding a time delay at line 66 of this file.
7) Go to http://localhost:8081 to see the envision visualization. 
    Refresh the browser, if simulation does not appear automatically.
8) An alternative way to visualize would be to use sumo-gui. Simply change line 46 of this file from 
    `sumo_headless=True` to `sumo_headless=False`. A sumo-gui will automatically pop up when the 
    simulation starts. A display is needed for the sumo-gui to work. 
"""
import gym
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.bezier_motion_planner import BezierMotionPlanner
from smarts.core.utils.episodes import episodes
from smarts.zoo.agent_spec import AgentSpec
import numpy as np

class ExampleAgent(Agent):
    def __init__(self, target_speed = 10):
        self.motion_planner = BezierMotionPlanner()
        self.target_speed = target_speed

    def act(self, obs):
        ego = obs.ego_vehicle_state
        current_pose = np.array([*ego.position[:2], ego.heading])

        # lookahead (at most) 10 waypoints
        target_wp = obs.waypoint_paths[0][:10][-1]
        dist_to_wp = target_wp.dist_to(obs.ego_vehicle_state.position)
        target_time = dist_to_wp / self.target_speed

        # Here we've computed the pose we want to hold given our target
        # speed and the distance to the target waypoint.
        target_pose_at_t = np.array(
            [*target_wp.pos, target_wp.heading, target_time]
        )

        # The generated motion planner trajectory is compatible
        # with the `ActionSpaceType.Trajectory`
        traj = self.motion_planner.trajectory(
            current_pose, target_pose_at_t, n=10, dt=0.5
        )
        return traj

AGENT_ID = "Agent-007"
agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Tracker),
    agent_params={"target_speed": 5},
    agent_builder=ExampleAgent
)

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/sumo/loop"],
    agent_specs={AGENT_ID: agent_spec},
)

for episode in episodes(n=100):
    agent = agent_spec.build_agent()
    observations = env.reset()
    episode.record_scenario(env.scenario_log)

    dones = {"__all__": False}
    while not dones["__all__"]:
        agent_obs = observations[AGENT_ID]
        action = agent.act(agent_obs)
        observations, rewards, dones, infos = env.step({AGENT_ID: action})
        episode.record_step(observations, rewards, dones, infos)

env.close()
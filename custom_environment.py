import gymnasium as gym
import numpy as np
from gymnasium import spaces
from multiprocessing import Pipe
from typing import Any


class FightingEnv(gym.Env):
    def __init__(self, pipe_environment_maestro: Pipe):
        super(FightingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.pipe_environment_maestro = pipe_environment_maestro
        self.action_space = spaces.Discrete(42)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 64, 96), dtype=np.uint8
        )
        self.own_hp = 0
        self.enemy_hp = 0
        self.last_observation = None
        self.terminated = False

    def step(self, action):
        message, data = None, None

        #print("wait for the next frame of fightingenv")
        if self.pipe_environment_maestro.poll():
            message, data = self.pipe_environment_maestro.recv()
            #print("trainings example received")

        if message == "TERMINATED":
            #print("episode terminated")
            self.terminated = True
            return None, 0, True, False, {}

        terminated = False

        # Execute one time step within the environment
        self.pipe_environment_maestro.send(["ACTION", action])
        message, data = self.pipe_environment_maestro.recv()
        if message == "OBSERVATION":
            observation = data
        elif message == "TERMINATED":
            self.terminated = True
            return None, 0, True, False, {}
        else:
            assert False
        message, data = self.pipe_environment_maestro.recv()
        if message == "FRAME_DATA":
            pass
        elif message == "TERMINATED":
            self.terminated = True
            return None, 0, True, False, {}
        else:
            assert False

        reward = 0
        if data["character_data"][0]["hp"] > self.enemy_hp:
            self.enemy_hp = data["character_data"][0]["hp"]
        if data["character_data"][1]["hp"] > self.own_hp:
            self.own_hp = data["character_data"][1]["hp"]
        if data["character_data"][0]["hp"] < self.enemy_hp:
            #print("enemy hp diff: ", (self.enemy_hp - data["character_data"][0]["hp"]))
            reward += (self.enemy_hp - data["character_data"][0]["hp"]) / 100
            self.enemy_hp = data["character_data"][0]["hp"]
        if data["character_data"][1]["hp"] < self.own_hp:
            #print("own hp diff: ", (self.own_hp - data["character_data"][1]["hp"]))
            reward += (data["character_data"][1]["hp"] - self.own_hp) / 100
            self.own_hp = data["character_data"][1]["hp"]

        if reward != 0:
            #print("reward: ", reward)
            pass

        return observation, reward, terminated, False, {}

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None,):
        """ Send a signal to maestro for restarting the game process. The newly created game will respond with a
        new initial observation (start of the game).

        Parameters
        ----------
        seed: ignored
        options: ignored
        """
        # Reset the state of the environment to an initial state
        if self.terminated:
            self.pipe_environment_maestro.send(["START_GAME_AGAIN", None])
            self.terminated = False
        else:
            self.pipe_environment_maestro.send(["RESET", None])

        # read initial observation
        observation = None
        message, data = self.pipe_environment_maestro.recv()
        if message == "OBSERVATION":
            observation = data
        else:
            assert False

        #print("first observation: ", observation)
        self.own_hp = 400
        self.enemy_hp = 400
        return (observation, None)

    def close(self):
        return super().close()

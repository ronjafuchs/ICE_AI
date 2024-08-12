import subprocess
from multiprocessing import Pipe, Process

from pyftg.gateway import Gateway
from stable_baselines3 import A2C
from custom_callback import PredictionCallback
from custom_environment import FightingEnv
from proxy_agent import ProxyAgent
from util import get_classpath_string

import time


def process_agent_e(training_pipe: Pipe, prediction_pipe: Pipe):
    prediction_callback = PredictionCallback(prediction_pipe)
    env = FightingEnv(training_pipe)
    print("env created")
    model = A2C("CnnPolicy", env, verbose=1)
    print("model created")
    model.learn(total_timesteps=100_000, callback=prediction_callback, progress_bar=True)
    print("model learned")
    model.save("a2c_fighting")


def run_player_game():
    return subprocess.Popen(
        get_classpath_string() + [
            "Main",
            "--limithp",
            "400",
            "400",
            "--inverted-player",
            "1",
            "--grpc-auto",
            "--port",
            "50060",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def game_player_communication():
    print("add two proxy agents and let them fight")
    gateway = Gateway(port=50060)
    gateway.register_ai("PROXY_AGENT", ProxyAgent())
    gateway.register_ai("PROXY_AGENT 2", ProxyAgent())
    gateway.run_game(["ZEN", "ZEN"], ["PROXY_AGENT", "PROXY_AGENT 2"], 2)


if __name__ == "__main__":
    # run a process and in which two players or agents can join to play
    # this one runs on port 50060
    game_player_process = run_player_game()
    pipe_proxy_agent_maestro, pipe_proxy_agent_worker = Pipe()  # not for reset
    time.sleep(1)

    # hooks into the previously opened game and connects two AI agents with it
    # adds two proxy agents, the pipe can be ignored for now
    player_game_comm = Process(
        target=game_player_communication
    )
    player_game_comm.start()


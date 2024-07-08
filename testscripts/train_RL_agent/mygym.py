import subprocess

import custom_environment
from stable_baselines3 import A2C
from multiprocessing import Pipe

from util import get_classpath_string


def run_train_game():
    return subprocess.Popen(
        get_classpath_string() +
        [
            "Main",
            "--limithp",
            "400",
            "400",
            "--inverted-player",
            "1",
            "--grpc-auto",
            "--port",
            "50051",
            "--fastmode",
            "-r",
            "20",
            "-f",
            "100000",
            "--disable-window",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    pipe_env_maestro, pipe_env_worker = Pipe()
    env = custom_environment.FightingEnv(pipe_env_worker)

    model = A2C("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save("a2c_fighting")

    """
    vec_env = model.get_env()
    obs = vec_env.reset()
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        #vec_env.render("human")
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()

    """
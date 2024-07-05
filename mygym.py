import custom_environment
from stable_baselines3 import A2C

if __name__ == "__main__":
    env = custom_environment.FightingEnv()

    model = A2C("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
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
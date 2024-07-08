# import autokeras as ak
# from keras.models import load_model
# from sklearn.preprocessing import LabelBinarizer
import subprocess
from multiprocessing import Pipe, Process
from multiprocessing.connection import wait

from pyftg.gateway import Gateway
from river import metrics
from river.forest import ARFClassifier
from river.imblearn import HardSamplingClassifier
from stable_baselines3 import A2C
from river.drift import ADWIN

from agent_e import AgentE
from agent_p import AgentP
from custom_callback import PredictionCallback
from custom_environment import FightingEnv
from proxy_agent import ProxyAgent
from util import flatten_dict, get_classpath_string


def process_agent_p(pipe: Pipe):
    metric = metrics.BalancedAccuracy()

    arf = ARFClassifier(
        n_models=25,
        metric=metric,
        seed=42,
        split_criterion="gini",
        grace_period=34,
        max_features=None,      # while not documented, None sets the amount of features used to all
        # available features (n_features)
        drift_detector=ADWIN(),
        lambda_value=20,
        leaf_prediction="nb",
        delta=0.12115012613326699,
        tau=0.8274041104298825,
    )
    predictor_model = HardSamplingClassifier(
        arf,
        size=30,
        p=0.036610084542451224,
        seed=42,
    )

    cm = metrics.ConfusionMatrix()
    metric_test = metrics.BalancedAccuracy(cm=cm)
    metric_test_two = metrics.Accuracy(cm=cm)

    learned = False
    while True:
        item = pipe.recv()
        message, data = item[0], item[1]
        if message == "PREDICT":
            prediction = None
            if not learned:
                prediction = "STAND"
            else:
                data.pop("character_data_0_action")
                prediction = predictor_model.predict_one(data)
            pipe.send(["OUTPUT_PREDICTION", [prediction]])
        elif message == "LEARN":
            data = flatten_dict(data)
            X = data.copy()
            y = X.pop("character_data_0_action")
            metric_test.update(y, predictor_model.predict_one(X))
            print(metric_test)
            print(metric_test_two)
            predictor_model.learn_one(X, y)
            learned = True


def process_agent_e(training_pipe: Pipe, prediction_pipe: Pipe):
    prediction_callback = PredictionCallback(prediction_pipe)
    env = FightingEnv(training_pipe)
    print("env created")
    model = A2C("CnnPolicy", env, verbose=1)
    print("model created")
    model.learn(total_timesteps=100_000, callback=prediction_callback)
    print("model learned")
    model.save("a2c_fighting")


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


def run_player_game():
    return subprocess.Popen(
        get_classpath_string() + [
            "Main",
            "--limithp",
            "400",
            "400",
            "--inverted-player",
            "1",
            "--grpc",
            "--port",
            "50060",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def game_train_communication(
    pipe_agent_p: Pipe, pipe_agent_e: Pipe, pipe_gateway: Pipe
):
    gateway = Gateway()
    gateway.register_ai("AgentP", AgentP(pipe_agent_p))
    gateway.register_ai("AgentE", AgentE(pipe_agent_e))
    gateway.run_game(["ZEN", "ZEN"], ["AgentP", "AgentE"], 1)


def game_player_communication(pipe_proxy_agent: Pipe):
    gateway = Gateway(port=50060)
    gateway.register_ai("PROXY_AGENT", ProxyAgent())
    gateway.register_ai("AgentE", AgentE(pipe_proxy_agent))
    gateway.run_game(["ZEN", "ZEN"], ["PROXY_AGENT", "AgentE"], 2)
    # gateway.run_game(["ZEN", "ZEN"], ["Human", "AgentE"], 2)


if __name__ == "__main__":
    game_process = run_train_game()
    game_player_process = run_player_game()
    pipe_agent_p_maestro, pipe_agent_p_game = Pipe()  # for reset
    pipe_agent_e_maestro, pipe_agent_e_game = Pipe()  # for reset
    pipe_env_maestro, pipe_env_worker = Pipe()  # not for reset
    pipe_prediction_maestro, pipe_prediction_worker = Pipe()  # not for reset
    pipe_proxy_agent_maestro, pipe_proxy_agent_worker = Pipe()  # not for reset
    pipe_gateway_maestro, pipe_gateway_worker = Pipe()  # for reset
    (
        pipe_agent_p_communication_maestro,
        pipe_agent_p_communication_worker,
    ) = Pipe()  # not for reset

    game_comm = Process(
        target=game_train_communication,
        args=(pipe_agent_p_game, pipe_agent_e_game, pipe_gateway_worker),
    )
    game_comm.start()
    player_game_comm = Process(
        target=game_player_communication, args=(pipe_proxy_agent_worker,)
    )
    player_game_comm.start()
    agent_e_worker = Process(
        target=process_agent_e,
        args=(
            pipe_env_worker,
            pipe_prediction_worker,
        ),
    )
    agent_e_worker.start()
    agent_p_worker = Process(
        target=process_agent_p, args=(pipe_agent_p_communication_worker,)
    )
    agent_p_worker.start()
    while True:
        connections_with_items = wait(
            [
                pipe_agent_p_maestro,
                pipe_agent_p_communication_maestro,
                pipe_agent_e_maestro,
                pipe_env_maestro,
                pipe_prediction_maestro,
                pipe_proxy_agent_maestro,
            ]
        )
        if pipe_agent_p_maestro in connections_with_items:
            message = pipe_agent_p_maestro.recv()
            pipe_agent_p_communication_maestro.send(message)
        if pipe_agent_p_communication_maestro in connections_with_items:
            message = pipe_agent_p_communication_maestro.recv()
            pipe_agent_p_maestro.send(message)
        if pipe_agent_e_maestro in connections_with_items:
            message = pipe_agent_e_maestro.recv()
            pipe_env_maestro.send(message)
        if pipe_env_maestro in connections_with_items:
            message = pipe_env_maestro.recv()
            if message[0] == "RESET":
                print("!resetting because it was requested from the env!")
                game_comm.terminate()
                game_process.terminate()
                game_process = run_train_game()
                pipe_agent_p_maestro, pipe_agent_p_game = Pipe()
                pipe_agent_e_maestro, pipe_agent_e_game = Pipe()
                pipe_gateway_maestro, pipe_gateway_worker = Pipe()
                game_comm = Process(
                    target=game_train_communication,
                    args=(pipe_agent_p_game, pipe_agent_e_game, pipe_gateway_worker),
                )
                game_comm.start()
            elif message[0] == "START_GAME_AGAIN":
                print("!starting game again because it was requested from the env!")
                game_comm.terminate()
                pipe_agent_p_maestro, pipe_agent_p_game = Pipe()
                pipe_agent_e_maestro, pipe_agent_e_game = Pipe()
                game_comm = Process(
                    target=game_train_communication,
                    args=(pipe_agent_p_game, pipe_agent_e_game, pipe_gateway_worker),
                )
                game_comm.start()
            else:
                pipe_agent_e_maestro.send(message)
        if pipe_prediction_maestro in connections_with_items:
            message = pipe_prediction_maestro.recv()
            if message[0] == "ACTION":
                pipe_proxy_agent_maestro.send(message)
            else:
                raise AssertionError()
        if pipe_proxy_agent_maestro in connections_with_items:
            message = pipe_proxy_agent_maestro.recv()
            if message[0] == "OBSERVATION":
                pipe_prediction_maestro.send(message)
            elif message[0] == "FRAME_DATA":
                new_message = ["LEARN", message[1]]
                pipe_agent_p_communication_maestro.send(new_message)

import autokeras as ak
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import subprocess
from multiprocessing import Pipe, Process
from multiprocessing.connection import wait
import pandas as pd
from pyftg.gateway import Gateway
from river import metrics
from stable_baselines3 import A2C
from agent_e import AgentE
from agent_p import AgentP
from custom_callback import PredictionCallback
from custom_environment import FightingEnv
from proxy_agent import ProxyAgent
from util import flatten_dict


def process_agent_p(pipe: Pipe):
    allowed_values = [
        "STAND",
        "FORWARD_WALK",
        "DASH",
        "BACK_STEP",
        "CROUCH",
        "JUMP",
        "FOR_JUMP",
        "BACK_JUMP",
        "STAND_GUARD",
        "CROUCH_GUARD",
        "AIR_GUARD",
        "THROW_A",
        "THROW_B",
        "STAND_A",
        "STAND_B",
        "CROUCH_A",
        "CROUCH_B",
        "AIR_A",
        "AIR_B",
        "AIR_DA",
        "AIR_DB",
        "STAND_FA",
        "STAND_FB",
        "CROUCH_FA",
        "CROUCH_FB",
        "AIR_FA",
        "AIR_FB",
        "AIR_UA",
        "AIR_UB",
        "STAND_D_DF_FA",
        "STAND_D_DF_FB",
        "STAND_F_D_DFA",
        "STAND_F_D_DFB",
        "STAND_D_DB_BA",
        "STAND_D_DB_BB",
        "AIR_D_DF_FA",
        "AIR_D_DF_FB",
        "AIR_F_D_DFA",
        "AIR_F_D_DFB",
        "AIR_D_DB_BA",
        "AIR_D_DB_BB",
        "STAND_D_DF_FC",
    ]

    label_binarizer = LabelBinarizer().fit(allowed_values)
    desired_columns = pd.read_csv("../../desired_columns.csv")
    predictor_model = load_model(
        "../../bigger_mctsai_model_autokeras", custom_objects=ak.CUSTOM_OBJECTS
    )

    cm = metrics.ConfusionMatrix()
    metric_test = metrics.BalancedAccuracy(cm=cm)
    metric_test_two = metrics.Accuracy(cm=cm)
    while True:
        item = pipe.recv()
        message, data = item[0], item[1]
        if message == "PREDICT":
            flattened_data = flatten_dict(data)
            df = pd.DataFrame(
                flattened_data,
                index=[0],
                columns=desired_columns.columns,
                dtype="str",
            )
            df = df.drop(columns=["character_data_0_action"])
            prediction = None
            prediction = predictor_model.predict(df, verbose=0)
            prediction = label_binarizer.inverse_transform(prediction)
            pipe.send(["OUTPUT_PREDICTION", [prediction]])
        elif message == "LEARN":
            flattened_data = flatten_dict(data)
            df = pd.DataFrame(
                flattened_data,
                index=[0],
                columns=desired_columns.columns,
                dtype="str",
            )
            X = df.drop(columns=["character_data_0_action"])
            y = df["character_data_0_action"]
            if y[0] not in allowed_values:
                continue
            y = label_binarizer.transform(y)
            predict = predictor_model.predict(X, verbose=0)
            metric_test.update(
                label_binarizer.inverse_transform(y)[0],
                label_binarizer.inverse_transform(predict)[0],
            )
            print(metric_test)
            print(metric_test_two)
            predictor_model.fit(X, y, verbose=0)


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
        [
            "java",
            "-XstartOnFirstThread",
            "-cp",
            "FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/macos/arm64/*:./lib/grpc/*",
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
        [
            "java",
            "-XstartOnFirstThread",
            "-cp",
            "FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/macos/arm64/*:./lib/grpc/*",
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

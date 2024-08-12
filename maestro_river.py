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

from agent_rl import AgentRL
from agent_river import AgentRiver
from custom_callback import PredictionCallback
from custom_environment import FightingEnv
from proxy_agent import ProxyAgent
from util import flatten_dict, get_classpath_string


def process_river_agent(pipe: Pipe):
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
            if not learned:
                prediction = "STAND"
                #print("model not yet learned, so we predict stand")
            else:
                data.pop("character_data_0_action")
                prediction = predictor_model.predict_one(data)

            pipe.send(["OUTPUT_PREDICTION", [prediction]])
        elif message == "LEARN":
            data = flatten_dict(data)
            #print("river example learn", data)

            X = data.copy()
            y = X.pop("character_data_0_action")
            metric_test.update(y, predictor_model.predict_one(X))
            #print(metric_test, metric_test_two)
            predictor_model.learn_one(X, y)
            learned = True


def process_rl_agent(pipe_environment: Pipe, pipe_prediction_requests: Pipe):
    prediction_callback = PredictionCallback(pipe_prediction_requests)
    env = FightingEnv(pipe_environment)
    #print("env created")
    model = A2C("CnnPolicy", env, verbose=1, device="cuda")
    #print("model created")
    model.learn(total_timesteps=100_000, callback=prediction_callback, progress_bar=True)
    #print("model learned")
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


def game_for_training_rl_agent_vs_river_agent(pipe_river_agent_to_maestro: Pipe, pipe_rl_agent_train_maestro: Pipe):
    gateway = Gateway(port=50051)
    gateway.register_ai("AgentRiver", AgentRiver(pipe_river_agent_to_maestro))
    gateway.register_ai("AgentRLTrain", AgentRL(pipe_rl_agent_train_maestro))
    gateway.run_game(["ZEN", "ZEN"], ["AgentRiver", "AgentRLTrain"], 1)


def game_play_against_rl_agent(pipe_rl_agent_maestro: Pipe):
    gateway = Gateway(port=50060)
    gateway.register_ai("PROXY_AGENT", ProxyAgent())
    gateway.register_ai("AgentRLApply", AgentRL(pipe_rl_agent_maestro))
    #gateway.run_game(["ZEN", "ZEN"], ["PROXY_AGENT", "AgentRLApply"], 2)
    gateway.run_game(["ZEN", "ZEN"], ["Human", "AgentRLApply"], 2)


if __name__ == "__main__":

    # the following categories exist:
    # maestro: the programm you are looking at right now, responsible for starting/stopping games and forwarding information to the other instances
    # messages often consist of two components, one is the description and the other the content
    # the description is used to decide where the message should be forwarded to
    # rl process: the rl model being trained, during training it will also make predictions for the rl agent
    # river process: the river model (behavior cloning) being trained, during training it will also make predictions for the river agent
    # rl agent apply: the rl agent used in fightinggameai to play against the human player, it will forward information to the RL process and request the action to apply
    # rl agent train: the rl agent used in fightinggameai to train against the river player, it will forward information to the RL process and request the action to apply
    # river agent: the river agent used in fightinggameai, it will forward information to the river process and request the action to apply
    # proxy agent: no idea what this is supposed to do yet
    # environment: the rl environment used for training, necessary to encode the game as a gymnasium environment for "simple" RL training

    ## pipe communication channels are either one-sided "x-to-y" or two-sided "x-back-and-forth-y"
    ## the first object names is the one having ownership of this pipe
    ## pipes always exist in pairs (x,y), if we send information into x it will be received by y, and vice-versa

    # region setup pipes
    ## two separate instances of rl agent (train and apply), the environment and maestro
    # sending information between the rl agent train (the agent training against the river agent) and maestro
    pipe_maestro_rl_agent_train, pipe_rl_agent_train_maestro = Pipe()  # for reset

    # sending information between the rl environment and maestro
    pipe_maestro_back_and_forth_environment, pipe_environment_back_and_forth_maestro = Pipe()  # not for reset

    # sending information between the rl process and maestro
    pipe_maestro_back_and_forth_rl_process, pipe_rl_process_back_and_forth_maestro = Pipe()  # not for reset

    # sending information between the rl agent apply (the agent playing against the human) and maestro
    pipe_maestro_rl_agent_apply, pipe_rl_agent_apply_maestro = Pipe()  # not for reset

    ## river agent, river process, maestro
    # sending information from maestro to the river agent
    pipe_maestro_to_river_agent, pipe_river_agent_to_maestro = Pipe()  # for reset
    # sending information between the behavior cloning river agent and maestro
    pipe_maestro_to_riverprocess, pipe_riverprocess_to_maestro = Pipe()  # not for reset
    #endregion

    # region start games
    # spawn the two processes for running games, supposed for the following combinations of agents
    process_training_game = run_train_game()             # training rl agent against the behavior cloning river agent
    process_player_game = run_player_game()     # human player, playing against the rl agent

    # start the training game, with the river agent and the rl agent
    process_communication_to_training_game = Process(
        target=game_for_training_rl_agent_vs_river_agent,
        args=(pipe_river_agent_to_maestro, pipe_rl_agent_train_maestro),
    )
    process_communication_to_training_game.start()

    # start the player's game, with the proxy agent and the rl agent
    process_communication_to_player_game = Process(
        target=game_play_against_rl_agent, args=(pipe_rl_agent_apply_maestro,)
    )
    process_communication_to_player_game.start()
    # endregion

    # region start processes for RL and river agent
    # run the rl model in a separate thread, it will later receive observations and requests for actions
    worker_rl_agent_training = Process(
        target=process_rl_agent,
        args=(
            pipe_environment_back_and_forth_maestro,
            pipe_rl_process_back_and_forth_maestro,
        ),
    )
    worker_rl_agent_training.start()
    
    # run the river agent in a separate thread
    worker_river_agent = Process(
        target=process_river_agent, args=(pipe_riverprocess_to_maestro,)
    )
    worker_river_agent.start()
    # endregion


    # this "while True" will ensure that information is forwarded in between processes indefinitely
    while True:
        # gets all the pipes that currently store information
        connections_with_items = wait(
            [
                pipe_maestro_to_river_agent,
                pipe_maestro_to_riverprocess,
                pipe_maestro_rl_agent_train,
                pipe_maestro_back_and_forth_environment,
                pipe_maestro_back_and_forth_rl_process,
                pipe_maestro_rl_agent_apply,
            ]
        )

        #region river agent application
        # the river agent requests actions from the riverprocess, forward information to the river process
        if pipe_maestro_to_river_agent in connections_with_items:
            message = pipe_maestro_to_river_agent.recv()
            pipe_maestro_to_riverprocess.send(message)

        # the river process responds with actions to apply, forward information to the river agent
        if pipe_maestro_to_riverprocess in connections_with_items:
            message = pipe_maestro_to_riverprocess.recv()
            pipe_maestro_to_river_agent.send(message)
        #endregion


        # if the rl agent apply previously requested an action, we will now receive it from the process
        # and forward it to the rl agent
        if pipe_maestro_back_and_forth_rl_process in connections_with_items:
            message = pipe_maestro_back_and_forth_rl_process.recv()
            if message[0] == "ACTION":
                pipe_maestro_rl_agent_apply.send(message)
            else:
                raise AssertionError()

        # conditional forwarding information from the rl agent to the RL or the River processes
        # Note, for every frame the proxy agent sends information to both
        # the RL process receives the image data, while the river process receives a dictionary of attributes of the game state
        if pipe_maestro_rl_agent_apply in connections_with_items:
            message = pipe_maestro_rl_agent_apply.recv()

            if message[0] == "OBSERVATION":
                # the rl agent to request an action
                pipe_maestro_back_and_forth_rl_process.send(message)

            elif message[0] == "FRAME_DATA":
                # the riveragent to learn from
                new_message = ["LEARN", message[1]]
                pipe_maestro_to_riverprocess.send(new_message)

        # the rl agent train sends information to the environment, the environment will react
        if pipe_maestro_rl_agent_train in connections_with_items:
            message = pipe_maestro_rl_agent_train.recv()
            pipe_maestro_back_and_forth_environment.send(message)

        # reaction of the environment
        if pipe_maestro_back_and_forth_environment in connections_with_items:
            message = pipe_maestro_back_and_forth_environment.recv()
            # three options:
            # (1) the game needs to be RESET while it is still running
            # (2) we want to START_GAME_AGAIN because it just finished
            # (3) the game continues normally, and we send the action to the rl train agent
            if message[0] == "RESET":
                print("!resetting because it was requested from the env!")
                # stop current agents and communication to training game
                process_communication_to_training_game.terminate()
                process_training_game.kill()

                # restart current agents and communication to training game
                process_training_game = run_train_game()
                pipe_maestro_to_river_agent, pipe_river_agent_to_maestro = Pipe()
                pipe_maestro_rl_agent_train, pipe_rl_agent_train_maestro = Pipe()
                process_communication_to_training_game = Process(
                    target=game_for_training_rl_agent_vs_river_agent,
                    args=(pipe_river_agent_to_maestro, pipe_rl_agent_train_maestro),
                )
                process_communication_to_training_game.start()

            elif message[0] == "START_GAME_AGAIN":
                print("!starting game again because it was requested from the env!")
                # here, the current game does not need to be stopped, because the API allows to restart at the end
                # of the game, we just need to reset the communication with the agents
                process_communication_to_training_game.terminate()
                pipe_maestro_to_river_agent, pipe_river_agent_to_maestro = Pipe()
                pipe_maestro_rl_agent_train, pipe_rl_agent_train_maestro = Pipe()
                process_communication_to_training_game = Process(
                    target=game_for_training_rl_agent_vs_river_agent,
                    args=(pipe_river_agent_to_maestro, pipe_rl_agent_train_maestro),
                )
                process_communication_to_training_game.start()

            else:
                pipe_maestro_rl_agent_train.send(message)

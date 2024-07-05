from river.multiclass import OneVsOneClassifier
from river.linear_model import LogisticRegression
from river import metrics
from time import process_time
from collections import deque
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from river.stream import Cache
import numpy as np
from river import stream
from river.preprocessing import OneHotEncoder
from river import compose
from river.compose import SelectType
from river.preprocessing import StandardScaler
from river.optim import SGD, Adam, RMSProp
from river.optim.losses import Hinge, Log
from river.optim.initializers import Zeros, Normal
from river.imblearn import HardSamplingClassifier

converter = {
    "character_data_0_hp": int,
    "character_data_1_hp": int,
    "character_data_0_x": int,
    "character_data_1_x": int,
    "character_data_0_y": int,
    "character_data_1_y": int,
    "character_data_0_left": int,
    "character_data_1_left": int,
    "character_data_0_right": int,
    "character_data_1_right": int,
    "character_data_0_top": int,
    "character_data_1_top": int,
    "character_data_0_bottom": int,
    "character_data_1_bottom": int,
    "character_data_0_action": str,
    "character_data_1_action": str,
    "character_data_0_front": bool,
    "character_data_1_front": bool,
    "character_data_0_attack_data_current_hit_area_left": int,
    "character_data_1_attack_data_current_hit_area_left": int,
    "character_data_0_attack_data_current_hit_area_right": int,
    "character_data_1_attack_data_current_hit_area_right": int,
    "character_data_0_attack_data_current_hit_area_top": int,
    "character_data_1_attack_data_current_hit_area_top": int,
    "character_data_0_attack_data_current_hit_area_bottom": int,
    "character_data_1_attack_data_current_hit_area_bottom": int,
    "front_0": bool,
    "front_1": bool,
    "character_data_0_speed_x": int,
    "character_data_1_speed_x": int,
    "character_data_0_speed_y": int,
    "character_data_1_speed_y": int,
    "character_data_0_attack_data_setting_hit_area_left": int,
    "character_data_1_attack_data_setting_hit_area_left": int,
    "character_data_0_attack_data_setting_hit_area_right": int,
    "character_data_1_attack_data_setting_hit_area_right": int,
    "character_data_0_attack_data_setting_hit_area_top": int,
    "character_data_1_attack_data_setting_hit_area_top": int,
    "character_data_0_attack_data_setting_hit_area_bottom": int,
    "character_data_1_attack_data_setting_hit_area_bottom": int,
    "character_data_0_attack_data_start_up": int,
    "character_data_1_attack_data_start_up": int,
    "character_data_0_attack_data_active": int,
    "character_data_1_attack_data_active": int,
    "character_data_0_attack_data_hit_damage": int,
    "character_data_1_attack_data_hit_damage": int,
    "character_data_0_attack_data_hit_add_energy": int,
    "character_data_1_attack_data_hit_add_energy": int,
    "character_data_0_attack_data_guard_add_energy": int,
    "character_data_1_attack_data_guard_add_energy": int,
    "character_data_0_attack_data_give_energy": int,
    "character_data_1_attack_data_give_energy": int,
    "character_data_0_attack_data_impact_x": int,
    "character_data_1_attack_data_impact_x": int,
    "character_data_0_attack_data_impact_y": int,
    "character_data_1_attack_data_impact_y": int,
    "character_data_0_attack_data_give_guard_recov": int,
    "character_data_1_attack_data_give_guard_recov": int,
    "character_data_0_attack_data_attack_type": int,
    "character_data_1_attack_data_attack_type": int,
    "character_data_0_attack_data_current_frame": int,
    "character_data_1_attack_data_current_frame": int,
    "character_data_0_energy": int,
    "character_data_1_energy": int,
    "character_data_0_hit_confirm": bool,
    "character_data_1_hit_confirm": bool,
    "character_data_0_hit_count": int,
    "character_data_1_hit_count": int,
    "character_data_0_last_hit_frame": int,
    "character_data_1_last_hit_frame": int,
    "character_data_0_state": str,
    "character_data_1_state": str,
}


def flatten_dict(d, parent_key="", sep="_"):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def objective(params):
    start = process_time()
    cm = metrics.ConfusionMatrix()
    acc = metrics.Accuracy(cm=cm)
    observation_stamps = []
    flatten_stamps = []
    print(params)
    window_size = params["window_size"]
    armf_params = params.copy()
    armf_params.pop("window_size")
    armf_params.pop("optimizer")
    reg_params = armf_params.pop("regularization")

    optimizer_type = params["optimizer"]["type"]
    if optimizer_type == "SGD":
        optimizer = SGD(lr=params["optimizer"]["learning_rate"])
    elif optimizer_type == "Adam":
        optimizer = Adam(
            lr=params["optimizer"]["learning_rate"],
            beta_1=params["optimizer"]["beta_1"],
            beta_2=params["optimizer"]["beta_2"],
            eps=params["optimizer"]["eps"],
        )
    elif optimizer_type == "RMSProp":
        optimizer = RMSProp(
            lr=params["optimizer"]["learning_rate"],
            rho=params["optimizer"]["rho"],
            eps=params["optimizer"]["eps"],
        )
    else:
        optimizer = None

    if reg_params["type"] == "l1":
        armf_params["l1"] = reg_params["l1_value"]
    elif reg_params["type"] == "l2":
        armf_params["l2"] = reg_params["l2_value"]

    sampling_params = armf_params.pop("sampler")
    one_hot_params = armf_params.pop("one_hot_params")
    select_str_bool = SelectType((str, bool))
    select_num = SelectType(int)
    classifier = OneVsOneClassifier(
        classifier=LogisticRegression(**armf_params, optimizer=optimizer)
    )
    classifier_wrapped = (
        HardSamplingClassifier(
            classifier=classifier,
            seed=42,
            size=sampling_params["size"],
            p=sampling_params["probability"],
        )
        if sampling_params["type"] == "hard_sampling"
        else classifier
    )
    model = (
        (
            (
                select_str_bool
                | OneHotEncoder(
                    drop_first=one_hot_params["drop_first"],
                    drop_zeros=one_hot_params["drop_zeros"],
                )
            )
            + select_num
        )
        | StandardScaler()
        | classifier_wrapped
    )
    learned_one = False
    window = None
    if window_size > 0:
        window = deque(maxlen=window_size)
    metric = metrics.BalancedAccuracy(cm=cm)
    for X, y in cache(X_y, key="big_data"):
        observation_stamp_start = process_time()
        observation_dict = {}
        if window is not None and len(window) > 0:
            observation_dict = {}
            for i in range(len(window)):
                observation_dict[f"observation_{i}"] = window[i]

        observation_dict["current_observation"] = X.copy()
        observation_stamps.append(process_time() - observation_stamp_start)

        # Flatten the observation_dict
        flatten_stamp_start = process_time()
        observation_dict = flatten_dict(observation_dict)
        flatten_stamps.append(process_time() - flatten_stamp_start)

        if learned_one:
            predict = model.predict_one(observation_dict)
            metric.update(y, predict)
        model.learn_one(observation_dict, y)
        if not learned_one:
            learned_one = True
            continue
        if window is not None:
            copy_dict = X.copy()
            copy_dict["character_data_0_action"] = y
            window.append(X)
    print(metric)
    print(acc)
    print(
        f"Observation stamps: {sum(observation_stamps)}, {max(observation_stamps)}, {min(observation_stamps)}, {sum(observation_stamps) / len(observation_stamps)}"
    )
    print(
        f"Flatten stamps: {sum(flatten_stamps)}, {max(flatten_stamps)}, {min(flatten_stamps)}, {sum(flatten_stamps) / len(flatten_stamps)}"
    )
    print(f"Total time: {process_time() - start}")
    return {"loss": 1 - metric.get(), "status": STATUS_OK}


random = np.random.default_rng(42)
cache = Cache()
X_y = stream.iter_csv(
    "/Users/gieseke/Bachelorarbeit/FightingICE_AI/big_data.csv",
    target="character_data_0_action",
    converters=converter,
)

space = {
    "window_size": hp.uniformint("window_size", 0, 30),
    "optimizer": hp.choice(
        "optimizer",
        [
            {
                "type": "SGD",
                "learning_rate": hp.loguniform("sgd_learning_rate", -7, -1),
            },
            {
                "type": "Adam",
                "learning_rate": hp.loguniform("adam_learning_rate", -7, -1),
                "beta_1": hp.uniform("adam_beta_1", 0.8, 0.999),
                "beta_2": hp.uniform("adam_beta_2", 0.9, 0.9999),
                "eps": hp.loguniform("adam_eps", -10, -1),
            },
            {
                "type": "RMSProp",
                "learning_rate": hp.loguniform("rmsprop_learning_rate", -7, -1),
                "rho": hp.uniform("rmsprop_rho", 0.8, 1.0),
                "eps": hp.loguniform("rmsprop_eps", -10, -1),
            },
        ],
    ),
    "loss": hp.choice("loss", [Log(), Hinge()]),
    "regularization": hp.choice(
        "regularization",
        [
            {"type": "None"},
            {"type": "l1", "l1_value": hp.loguniform("l1_value", -8, 1)},
            {"type": "l2", "l2_value": hp.loguniform("l2_value", -8, 1)},
        ],
    ),
    "intercept_init": hp.uniform("intercept_init", -1, 1),
    "intercept_lr": hp.loguniform("intercept_lr", -8, 0),
    "clip_gradient": hp.loguniform("clip_gradient", 0, 30),
    "sampler": hp.choice(
        "sampler",
        [
            {"type": "None"},
            {
                "type": "hard_sampling",
                "size": hp.uniformint("sampler_size", 1, 100),
                "probability": hp.uniform("sampler_probability", 0, 1),
            },
        ],
    ),
    "one_hot_params": {
        "drop_first": hp.choice("drop_first", [True, False]),
        "drop_zeros": hp.choice("drop_zeros", [True, False]),
    },
}
best = fmin(objective, space, algo=tpe.suggest, max_evals=1000, rstate=random)
print(best)

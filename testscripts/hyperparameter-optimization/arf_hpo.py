from collections import deque
import numpy as np
from river import stream
from river.forest import ARFClassifier
from river import metrics
from river.stream import Cache
from time import process_time
from river.imblearn import HardSamplingClassifier
from river.drift import DriftRetrainingClassifier

from hyperopt import hp, fmin, tpe, STATUS_OK

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
    arf_params = params.copy()
    arf_params.pop("window_size")
    sampler_params = arf_params.pop("sampler")
    classifier = ARFClassifier(
        **arf_params, stop_mem_management=True, memory_estimate_period=1000
    )
    drift_classifier = DriftRetrainingClassifier(classifier=classifier)
    model = HardSamplingClassifier(
        classifier=drift_classifier,
        size=sampler_params["size"],
        p=sampler_params["probability"],
        seed=42,
    )
    window = None
    if window_size > 0:
        window = deque(maxlen=window_size)
    metric = metrics.BalancedAccuracy(cm=cm)
    trained = False
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

        if trained:
            predict = model.predict_one(observation_dict)
            metric.update(y, predict)
        model.learn_one(observation_dict, y)
        trained = True
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


space = {
    "n_models": hp.uniformint("n_models", 10, 50),
    "grace_period": hp.uniformint("grace_period", 0, 50),
    "max_features": None,
    "lambda_value": hp.uniformint("lambda_value", 6, 30),
    "split_criterion": hp.choice("split_criterion", ["gini", "info_gain", "hellinger"]),
    "delta": hp.uniform("delta", 0, 1),
    "tau": hp.uniform("tau", 0, 1),
    "leaf_prediction": hp.choice("leaf_prediction", ["mc", "nb", "nba"]),
    "binary_split": True,
    "seed": 42,
    "metric": hp.choice(
        "metric",
        [
            metrics.BalancedAccuracy(),
        ],
    ),
    "window_size": hp.uniformint("window_size", 0, 5),
    "max_size": 4000,
    "sampler": {
        "size": hp.uniformint("sampler_size", 1, 50),
        "probability": hp.uniform("sampler_probability", 0, 1),
    },
}

for key, value in space.items():
    print(f"Range for {key}: {value}")


random = np.random.default_rng(42)
cache = Cache()
X_y = stream.iter_csv(
    "/Users/gieseke/Bachelorarbeit/FightingICE_AI/big_data.csv",
    target="character_data_0_action",
    converters=converter,
)
best = fmin(objective, space, algo=tpe.suggest, max_evals=1000, rstate=random)

print(best)

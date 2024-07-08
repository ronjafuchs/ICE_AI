from river.forest import AMFClassifier
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
    select_str_bool = SelectType((str, bool))
    select_num = SelectType(int)
    classifier = AMFClassifier(**armf_params)
    model = ((select_str_bool | OneHotEncoder()) + select_num) | classifier
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
        learned_one = True
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
    "window_size": hp.uniformint("window_size", 0, 10),
    "n_estimators": hp.uniformint("n_estimators", 1, 25),
    "dirichlet": hp.uniform("dirichlet", 0, 0.5),
    "split_pure": hp.choice("split_pure", [True, False]),
    "step": hp.uniform("step", 0, 2),
    "seed": 42,
    "use_aggregation": hp.choice("use_aggregation", [True, False]),
}
best = fmin(objective, space, algo=tpe.suggest, max_evals=1000, rstate=random)
print(best)

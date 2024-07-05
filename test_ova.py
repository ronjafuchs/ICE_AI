from river import ensemble, datasets, metrics, evaluate, stream
from river.forest import ARFClassifier

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


# Load a multi-class dataset
dataset = stream.iter_csv(
    "/Users/gieseke/Bachelorarbeit/FightingICE_AI/big_data.csv", target="character_data_0_action", converters=converter,
    )

# Identify unique classes in the dataset
unique_classes = set(y for x, y in dataset)

# Create an ARF model for each class
arf_models = {}
for cls in unique_classes:
    arf_models[cls] = ARFClassifier(seed=42)

# Initialize metrics
metric = metrics.BalancedAccuracy()

# Train and evaluate each ARF model
dataset = stream.iter_csv(
    "/Users/gieseke/Bachelorarbeit/FightingICE_AI/big_data.csv", target="character_data_0_action", converters=converter,
)
for x, y in dataset:
    prediction = []
    for cls, model in arf_models.items():
        if model._n_samples_seen < 1:
            model.learn_one(x, (y == cls))
            continue
        predic = model.predict_proba_one(x)
        model.learn_one(x, (y == cls))
        if True in predic.keys():
            prediction.append((cls,predic.get(True)))
    if len(prediction) == 0:
        continue
    print(prediction)
    prediction = max(prediction, key=lambda x: x[1])[0]
    print(y, prediction)
    metric.update(y, prediction)
    prediction = []
    print(metric)


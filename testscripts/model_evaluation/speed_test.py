from river.stream import Cache
from river import stream
from time import process_time
import statistics

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
# Initialize Cache and Stream
cache = Cache()
X_y = stream.iter_csv(
    "/Users/gieseke/Bachelorarbeit/FightingICE_AI/big_data.csv",
    target="character_data_0_action",
    converters=converter,
)

# Variables to store time durations
first_pass_cache_times = []
second_pass_cache_times = []
no_cache_times = []

# Number of runs
num_runs = 100

# Multiple runs for First pass with cache
for i in range(num_runs):
    try:
        cache.clear("kek")
    except FileNotFoundError:
        pass
    X_y = stream.iter_csv(
        "/Users/gieseke/Bachelorarbeit/FightingICE_AI/big_data.csv",
        target="character_data_0_action",
        converters=converter,
    )
    start = process_time()
    for X, y in cache(X_y, key="kek"):
        pass
    end = process_time()
    first_pass_cache_times.append(end - start)

# Multiple runs for Second pass with cache
for i in range(num_runs):
    X_y = stream.iter_csv(
        "/Users/gieseke/Bachelorarbeit/FightingICE_AI/big_data.csv",
        target="character_data_0_action",
        converters=converter,
    )
    start = process_time()
    for X, y in cache(X_y, key="kek"):
        pass
    end = process_time()
    second_pass_cache_times.append(end - start)

# Multiple runs for No cache
for i in range(num_runs):
    X_y = stream.iter_csv(
        "/Users/gieseke/Bachelorarbeit/FightingICE_AI/big_data.csv",
        target="character_data_0_action",
        converters=converter,
    )
    start = process_time()
    for X, y in X_y:
        pass
    end = process_time()
    no_cache_times.append(end - start)

# Calculate and print statistics also print the sum
print(
    "First pass with cache: Sum={}, Min={}, Max={}, Avg={}".format(
        sum(first_pass_cache_times),
        min(first_pass_cache_times),
        max(first_pass_cache_times),
        statistics.mean(first_pass_cache_times),
    )
)
print(
    "Second pass with cache: Sum={}, Min={}, Max={}, Avg={}".format(
        sum(second_pass_cache_times),
        min(second_pass_cache_times),
        max(second_pass_cache_times),
        statistics.mean(second_pass_cache_times),
    )
)
print(
    "No cache: Sum={}, Min={}, Max={}, Avg={}".format(
        sum(no_cache_times),
        min(no_cache_times),
        max(no_cache_times),
        statistics.mean(no_cache_times),
    )
)

import platform
from typing import List


def flatten_dict(d, parent_key='', sep='_'):
        flattened_dict = {}
        for key, value in d.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, dict):
                flattened_dict.update(flatten_dict(value, new_key, sep))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    index_key = f"{new_key}{sep}{i}"
                    if isinstance(item, dict):
                        flattened_dict.update(flatten_dict(item, index_key, sep))
                    else:
                        flattened_dict[index_key] = item
            else:
                flattened_dict[new_key] = value
        return flattened_dict


def get_classpath_string() -> List:
    system = platform.system()
    if system == 'Darwin':
        return [
            "java",
            "-XstartOnFirstThread",
            "-cp",
            "FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/macos/arm64/*:./lib/grpc/*"
        ]
    if system == 'Linux':
        return [
            "java",
            "-cp",
            "FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/linux/amd64/*:./lib/grpc/*"
        ]

    if system == 'Windows':
        return [
            "java",
            "-cp",
            "FightingICE.jar;./lib/*;./lib/lwjgl/*;./lib/lwjgl/natives/windows/amd64/*;./lib/grpc/*;"
        ]
    raise ValueError(f"system needs to be of type 'Darwin' (Mac), Linux, or Windows but {system} has been found. "
                     f"Add run-script command in util.py to support your OS")

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
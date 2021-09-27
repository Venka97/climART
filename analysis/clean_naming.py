def get_model_name(name: str) -> str:
    if 'GN' in name:
        return 'GraphNet'
    elif 'CNN' in name:
        return 'CNN'
    elif 'MLP' in name:
        return 'MLP'
    elif 'GCN' in name:
        return 'GCN'
    else:
        raise ValueError(name)
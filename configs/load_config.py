import yaml

def load_configs(yaml_path="config.yaml"):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config
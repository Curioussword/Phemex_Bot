import yaml

import os



def load_config():

    """Load the configuration file."""

    try:

        with open(os.path.join(os.path.dirname(__file__), 'config', 'config.yaml'), 'r') as file:

            config = yaml.safe_load(file)

        return config

    except FileNotFoundError:

        print("Configuration file not found. Please check the path.")

        return None

    except yaml.YAMLError as e:

        print(f"Error parsing YAML: {e}")

        return None
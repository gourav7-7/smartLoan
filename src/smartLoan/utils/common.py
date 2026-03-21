import os
from box.exceptions import BoxValueError
import yaml
from smartLoan.utils import logger
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """read yaml file and returns
    Args - path like input 
    raises: valueerror if yaml file is empty
    returns: ConfigBox type obj"""

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """creates list of directories
    args: path_to_directories (list)- list of path of directories
    ignore_log (bool,optional): ignore if multiple dirs is to be created"""

    for path in path_to_directories:
        os.mkdir(path, exist_ok=True)
        if verbose:
            logger.info(f"created directories at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    args:
        path - path to json file
        data - data to be saved in json"""
    
    with open(path, "w") as f:
        json.dump(data, f, indent = 4)
    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data
    
    args: path - path to json file
    
    returns: configbox - data as class attributes instead of dict"""

    with open(path) as f:
        content = json.load(f)
    
    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

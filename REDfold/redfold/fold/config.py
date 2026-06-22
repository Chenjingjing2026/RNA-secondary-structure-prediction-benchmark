import json
import os
import munch
import random
import numpy as np
from shutil import copyfile


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict


def process_config(jsonfile):
    try:
      config_dict = get_config_from_json(jsonfile)
    except:
      raise SystemExit('error: Fail in configuration!')
    
    config = munch.Munch(config_dict)
    
    return config

def set_config(file1,file2):
    try:
      copyfile(file1, file2)
    except:
      raise SystemExit('error: Fail in configuration!')




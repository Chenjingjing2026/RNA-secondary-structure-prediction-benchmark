import os
import sys
from argparse import ArgumentParser
from .fold import *
#from .fold.utils import *
#from .fold.config import *
from .process_data import process_data
from .test_data import test_data

redfold_path= os.path.join(os.path.dirname(__file__), 'data')
redfold_config= os.path.join(os.path.dirname(__file__), 'data', 'config.json')
default_path= os.path.join('./', 'data')
default_config= os.path.join('./', 'data', 'config.json')


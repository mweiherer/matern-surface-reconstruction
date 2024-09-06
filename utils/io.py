import os
import yaml
import json
from pathlib import Path


def cond_mkdir(path):   
    Path(path).mkdir(parents = True, exist_ok = True)

#########################################################################################
# Code to add an '!include' constructor to yaml. Code has been adapted from
# https://gist.github.com/joshbode/569627ced3076931b02f.
class Loader(yaml.SafeLoader):
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader, node):
    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())
        
yaml.add_constructor('!include', construct_include, Loader)
#########################################################################################
 
def read_config(config):
    '''
    Loads a given config file in yaml format.
    :param config: The config file in yaml format
    :return: The config file as dictionary
    '''
    with open(config, 'r') as stream:
        cfg = yaml.load(stream, Loader)
    return cfg

def decode_type(value):
    '''
    Converts an input string to its respective data type. Supports boolean values, 
    lists of integers and floats, and integers and floats. If none of the data types
    match the input, string is returned. 
    :param value: The input string
    :return: The input string converted to its respective data type
    '''
    # Check if input string is a bool.
    if value.lower() == 'true': return True
    if value.lower() == 'false': return False

    # Check if input string is a list (of integers or floats).
    if value[0] == '[' and value[-1] == ']':
        values = value.strip('][').strip().split(',')

        try:
            return [int(value) for value in values]
        except: pass

        try:
            return [float(value) for value in values]
        except:
            return [value.strip() for value in values]

    # Check if input string is an integer.
    try:
        return int(value)
    except: pass

    # Check if input string is a float.
    try:
        return float(value)
    except:
        return value # If it's none of the above, then return string.
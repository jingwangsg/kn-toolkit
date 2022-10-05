import importlib
import os
from util.general import import_modules

cur_dir = os.path.dirname(__file__)
import_modules(cur_dir, "util.data")
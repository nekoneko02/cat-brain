import importlib
from . import cat
from . import toy
from . import dummy
importlib.reload(cat)
importlib.reload(toy)
importlib.reload(dummy)
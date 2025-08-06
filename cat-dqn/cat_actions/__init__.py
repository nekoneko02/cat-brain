import importlib
from . import chase
from . import stop
from . import escape
importlib.reload(chase)
importlib.reload(escape)
importlib.reload(stop)

# class名とクラスオブジェクトのマッピング
cat_actions_mapping = {
    "Chase": chase.Chase,
    "Stop": stop.Stop,
    "Escape": escape.Escape,
}
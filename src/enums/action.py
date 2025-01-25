from enum import Enum, unique


@unique
class Action(Enum):
    Dig = 0
    Pass = 4
    Serve = 3
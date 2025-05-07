from enum import Enum

TIMESTEP = 0.025


class Direction:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class RoadPosition(Enum):
    UP_OF_ROAD = 0
    DOWN_OF_ROAD = 1
    LEFT_OF_ROAD = 2
    RIGHT_OF_ROAD = 3
    ERROR_ARISE = -1


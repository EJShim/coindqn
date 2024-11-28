import random


class Player:
    def __init__(self):
        self._my_number = None
        self._column = None
        self._row = None

    def get_name(self) -> str:

        return "python player"

    def initialize(self, my_number: int, column: int, row: int):
        
        self._my_number = my_number
        self._column = column
        self._row = row

    def move_next(self, map: list[int], my_position: int) -> int:

        direction = random.randint(0, 3)
        return direction
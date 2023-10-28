from enum import Enum


class Action(Enum):
    Sell = -1
    Hold = 0
    Buy = 1

    def __eq__(self, __value: object) -> bool:
        return self.value == __value

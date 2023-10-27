from enum import Enum
from dataclasses import dataclass


class ActionType(Enum):
    Sell = -1
    Hold = 0
    Buy = 1

    def __eq__(self, __value: object) -> bool:
        return self.value == __value


@dataclass
class Action:
    # Properties (public)
    type: ActionType
    quantity: int

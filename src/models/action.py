from dataclasses import dataclass
from enums import Action as ActionType


@dataclass
class Action:
    # Properties (public)
    type: ActionType
    quantity: int

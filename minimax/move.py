from dataclasses import dataclass
from enum import Enum


class Player(Enum):
    P1 = "X"  # Min
    P2 = "O"  # Max


@dataclass
class Move:
    """Move class represents a single move in tictactoe game"""

    row: int
    col: int
    player: Player
    score: int = 0

    def __eq__(self, other: "Move") -> bool:
        return (
            self.row == other.row
            and self.col == other.col
            and self.player == other.player
        )

from typing import List, Optional
from enum import Enum
from move import Player, Move
from exceptions import InvalidMove


class Result(Enum):
    ONGOING = "ONGOING"
    P1_WIN = "PLAYER 1 WINS!"
    P2_WIN = "PLAYER 2 WINS!"
    DRAW = "DRAW..."


Square = Optional[Player]


class Board:
    """Board class is responsible for representing tictactoe game state"""

    def __init__(
        self,
        size: int = 3,
        win_condition: int = 3,
        next_player: Player = Player.P1,
        squares: Optional[List[List[Square]]] = None,
    ) -> None:
        self.size: int = size
        self.win_condition: int = win_condition  # How many consecutive squares to win

        self.score = 0
        self.result: Result = Result.ONGOING
        self.next_player: Player = next_player
        self.squares: List[List[Square]] = (
            squares
            if squares != None
            else [[None for _ in range(size)] for _ in range(size)]
        )

    def make_move(self, move: Move) -> None:
        if move.row >= self.size or move.col >= self.size:
            raise InvalidMove("Out of bounds!")

        if not self.is_empty(move.row, move.col):
            raise InvalidMove("Non empty field!")

        if move.player != self.next_player:
            raise InvalidMove(f"Current player is '{self.next_player}'!")

        self.squares[move.row][move.col] = move.player
        self._next_player()

    def undo_move(self, move: Move) -> None:
        self.squares[move.row][move.col] = None
        self.result = Result.ONGOING
        self._next_player()

    def is_empty(self, row: int, col: int) -> bool:
        if self.squares[row][col] == None:
            return True
        return False

    def _next_player(self) -> None:
        self.next_player = Player.P2 if self.next_player == Player.P1 else Player.P1

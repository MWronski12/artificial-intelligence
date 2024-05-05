from board import Board
from move import Move
from typing import List


class Gui:
    def __init__(self) -> None:
        raise NotImplementedError("Abstract class Gui")

    def draw(self) -> None:
        raise NotImplementedError("Abstract class Gui")

    def get_player_move(self) -> Move:
        raise NotImplementedError("Abstract class Gui")


class ConsoleGui(Gui):
    """Console Gui can draw a tictactoe game in console and capture human player move decision"""

    def __init__(self, board: Board) -> None:
        self.board: Board = board

    def draw(self) -> None:
        for row in self.board.squares:
            print([square.value if square != None else " " for square in row])
        print()

    def get_player_move(self) -> Move:
        while True:
            player_input: str = input(
                f"Enter row, column of Your move [example input '2 1']: "
            )

            try:
                input_list: List[str] = player_input.split()
                row: int = int(input_list[0])
                col: int = int(input_list[1])
                move: Move = Move(row, col, self.board.next_player)

                if not self.board.is_empty(move.row, move.col):
                    print("This square is not empty!")
                    continue

                return move

            except:
                print("Invalid move input!")

from board import Board, Result

from typing import Dict


class Evaluate:
    """Evaluate class provides method for evaluating tictactoe game state"""

    result_score_map: Dict[Result, int] = {
        Result.P1_WIN: -1,
        Result.P2_WIN: 1,
        Result.ONGOING: 0,
        Result.DRAW: 0,
    }

    @staticmethod
    def eval(board: Board) -> int:
        return Evaluate.result_score_map[board.result]

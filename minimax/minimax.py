from board import Board, Result
from valid_moves import ValidMoves
from evaluate import Evaluate
from move import Move, Player
from exceptions import GameOver
from stats import Stats

from typing import List

import time


class MiniMax:

    @staticmethod
    def get_best_move(board: Board, stats: Stats = None) -> Move:
        start = time.time()
        moves: List[Move] = ValidMoves.get_valid_moves(board)
        if board.result != Result.ONGOING:
            raise GameOver("No moves available!")

        for move in moves:
            board.make_move(move)
            move.score = MiniMax.minimax(board, 1, stats)
            board.undo_move(move)

        key = lambda move: move.score if board.next_player == Player.P2 else -move.score
        if stats != None:
            stats.execution_time = time.time() - start
        return max(moves, key=key)

    @staticmethod
    def minimax(board: Board, depth: int, stats: Stats = None) -> int:
        if stats != None:
            stats.nodes_visited += 1

        moves: List[Move] = ValidMoves.get_valid_moves(board)
        if len(moves) == 0:
            if stats != None:
                stats.aggregated_depth += depth
                stats.leaf_count += 1
            return Evaluate.eval(board)

        for move in moves:
            board.make_move(move)
            move.score = MiniMax.minimax(board, depth + 1, stats)
            board.undo_move(move)

        key = lambda move: move.score if board.next_player == Player.P2 else -move.score

        return max(moves, key=key).score

from board import Board
from move import Player, Move
from evaluate import Evaluate
from valid_moves import ValidMoves
from exceptions import GameOver
from stats import Stats

from typing import List

import math
import time


class AlphaBeta:
    @staticmethod
    def get_best_move(board: Board, stats: Stats = None) -> Move:
        start = time.time()
        moves: List[Move] = ValidMoves.get_valid_moves(board)
        if len(moves) == 0:
            raise GameOver("No moves available")

        for move in moves:
            board.make_move(move)
            move.score = AlphaBeta.alpha_beta(
                board=board,
                alpha=-math.inf,
                beta=math.inf,
                depth=1,
                max_player=board.next_player == Player.P2,
                stats=stats,
            )
            board.undo_move(move)

        key = lambda move: move.score if board.next_player == Player.P2 else -move.score
        if stats != None:
            stats.execution_time = time.time() - start
        return max(moves, key=key)

    @staticmethod
    def alpha_beta(
        board: Board,
        alpha: int,
        beta: int,
        depth: int,
        max_player: bool,
        stats: Stats = None,
    ) -> int:
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
            move.score = AlphaBeta.alpha_beta(
                board, alpha, beta, depth + 1, not max_player, stats
            )
            board.undo_move(move)

            if max_player:
                alpha = max(move.score, alpha)
            else:
                beta = min(move.score, beta)

            if beta <= alpha:
                        if stats != None:
                            stats.nodes_pruned += 1
                        break

        key = lambda move: move.score if max_player else -move.score
        return max(moves, key=key).score

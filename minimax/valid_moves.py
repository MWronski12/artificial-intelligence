from board import Board, Result
from move import Move, Player

from typing import List, Callable


class ValidMoves:

    @staticmethod
    def get_valid_moves(board: Board) -> List[Move]:
        if ValidMoves.game_is_over(board):
            return []

        player: Player = board.next_player
        moves: List[Move] = []

        for row in range(board.size):
            for col in range(board.size):
                if board.is_empty(row, col):
                    move = Move(row, col, player)
                    moves.append(move)

        return moves

    @staticmethod
    def game_is_over(board: Board) -> bool:

        def wrapper(
            func: Callable[[Board], Result],
            board: Board,
        ) -> Result:
            return func(board) if board.result == Result.ONGOING else board.result

        board.result = wrapper(ValidMoves._check_horizontal_rays, board)
        board.result = wrapper(ValidMoves._check_vertical_rays, board)
        board.result = wrapper(ValidMoves._check_diagonal_rays, board)
        board.result = wrapper(ValidMoves._check_antidiagonal_rays, board)
        board.result = wrapper(ValidMoves._check_draw, board)

        return board.result != Result.ONGOING

    @staticmethod
    def _check_horizontal_rays(board: Board) -> Result:
        for row in board.squares:
            for i in range(board.size - board.win_condition + 1):
                if row[i] != None and all(
                    row[i + j] == row[i] for j in range(1, board.win_condition)
                ):
                    return ValidMoves._player_to_result(row[i])

        return Result.ONGOING

    @staticmethod
    def _check_vertical_rays(board: Board) -> Result:
        for col in range(board.size):
            for i in range(board.size - board.win_condition + 1):
                if board.squares[i][col] != None and all(
                    board.squares[i + k][col] == board.squares[i][col]
                    for k in range(1, board.win_condition)
                ):
                    return ValidMoves._player_to_result(board.squares[i][col])

        return Result.ONGOING

    @staticmethod
    def _check_diagonal_rays(board: Board) -> Result:
        for i in range(board.size - board.win_condition + 1):
            for j in range(board.size - board.win_condition + 1):
                if board.squares[i][j] != None and all(
                    board.squares[i + k][j + k] == board.squares[i][j]
                    for k in range(1, board.win_condition)
                ):
                    return ValidMoves._player_to_result(board.squares[i][j])

        return Result.ONGOING

    @staticmethod
    def _check_antidiagonal_rays(board: Board) -> Result:
        for i in range(board.size - board.win_condition + 1):
            for j in range(board.win_condition - 1, board.size):
                if board.squares[i][j] != None and all(
                    board.squares[i + k][j - k] == board.squares[i][j]
                    for k in range(1, board.win_condition)
                ):
                    return ValidMoves._player_to_result(board.squares[i][j])

        return Result.ONGOING

    @staticmethod
    def _check_draw(board: Board) -> Result:
        for row in board.squares:
            for square in row:
                if square == None:
                    return Result.ONGOING

        return Result.DRAW

    @staticmethod
    def _player_to_result(player: Player) -> Result:
        if player not in [Player.P1, Player.P2]:
            raise RuntimeError("Logic error!")

        return Result.P1_WIN if player == Player.P1 else Result.P2_WIN

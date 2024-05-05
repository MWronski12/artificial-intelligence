from board import Board
from gui import ConsoleGui
from move import Player
from alphabeta import AlphaBeta
from valid_moves import ValidMoves


def main():
    board = Board()
    gui = ConsoleGui(board)
    gui.draw()

    while True:
        if board.next_player == Player.P1:
            move = gui.get_player_move()
        elif board.next_player == Player.P2:
            move = AlphaBeta.get_best_move(board)

        board.make_move(move)
        gui.draw()

        if ValidMoves.game_is_over(board):
            print(board.result)
            return


if __name__ == "__main__":
    main()

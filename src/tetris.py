from enum import IntEnum
from typing import List

import numpy as np

import DrawingHelper
from Mino import Mino, MinoType


class TetrisAction(IntEnum):
    Nothing = 0
    RotateClockwise = 1
    RotateCounterclockwise = 2
    Left = 3
    Right = 4


class Tetris:
    def __init__(self):
        self.init()

    def init(self):
        """Initialize all the components of the game"""
        self.board = self.init_board()
        self.minos = self.init_minos()
        self.score = self.init_score()
        self.rate = self.init_rate()
        self.chain = self.init_chain()
        self.canvas = self.init_canvas()

    def init_board(self):
        """Create a new empty board that is initially filled with zeros.
        The left, right and bottom edges are filled with 1.
        """
        board = np.zeros((25, 16))
        board[:, 2] = 1
        board[:, -3] = 1
        board[21, :] = 1
        return board

    def init_minos(self) -> List[Mino]:
        """Return a list of 5 random minos, initialized at position (5, 0)

        Returns:
            List[Mino]: mino list
        """
        return [Mino(5, 0, MinoType(np.random.randint(7))) for _ in range(5)]

    def init_score(self) -> int:
        """Return the initial score

        Returns:
            int: initial score
        """
        return 0

    def init_rate(self) -> float:
        """Return the initial rate (speed at which game goes)

        Returns:
            float: initial rate
        """
        return 1.0

    def init_chain(self) -> int:
        """Return the initial chain (number of removed lines)

        Returns:
            int: initial chain
        """
        return 0

    def init_canvas(self):
        return DrawingHelper.draw(self.board[:22, 2:-2], self.minos, self.score)

    def advance_one_step(self, action: TetrisAction):
        """Advance the game by one step.
        Will do the given action, if any, and if the current mino did not reach the bottom yet lower
        it by one cell.
        If the game is over, reset it.

        Args:
            action (TetrisAction): the action to do
        """

        is_dead = self.update_mino(action)

        self.canvas = DrawingHelper.draw(self.board[:22, 2:-2], self.minos, self.score)

        if is_dead:
            self.init()

    def refresh_canvas(self):
        self.canvas = DrawingHelper.draw(self.board[:22, 2:-2], self.minos, self.score)

    def update_mino(self, action: TetrisAction) -> bool:
        """Make one step of the game.
        Will do the given action, if any, and if the current mino did not reach the bottom yet lower
        it by one cell. If the bottom / another mino is below the current mino, check and return
        whether the game is over.

        Args:
            action (TetrisAction): the action to do

        Returns:
            bool: if the game is over
        """
        # make a temporal copy of the current mino to test collisions
        tmp_mino = self.minos[0].copy()

        # if there is an action we have to do, do it
        if action == TetrisAction.RotateClockwise:
            tmp_mino.mino = np.rot90(tmp_mino.mino)
        elif action == TetrisAction.RotateCounterclockwise:
            tmp_mino.mino = np.rot90(tmp_mino.mino, -1)
        elif action == TetrisAction.Right:
            tmp_mino.x += 1
        elif action == TetrisAction.Left:
            tmp_mino.x -= 1

        if not tmp_mino.collision(self.board):
            self.minos[0].x = tmp_mino.x
            self.minos[0].y = tmp_mino.y
            self.minos[0].mino = tmp_mino.mino

        tmp_mino = self.minos[0].copy()

        # go down one step
        tmp_mino.y += 1

        if tmp_mino.collision(self.board):
            # put the next mino at the top of the board
            self.update_board(self.minos[0])

            # advance the backlog of minos by one
            self.minos = self.minos[1:]

            # create a new random mino and append to backlog
            self.minos.append(Mino(5, 0, MinoType(np.random.randint(7))))

        else:
            # update the minos position
            self.minos[0].x = tmp_mino.x
            self.minos[0].y = tmp_mino.y
            self.minos[0].mino = tmp_mino.mino

        return self.check_dead()

    def update_board(self, mino):
        h, w = mino.mino.shape
        self.board[mino.y : mino.y + h, mino.x : mino.x + w] += mino.mino
        self.check_line()

    def check_line(self):
        tmp = self.board[:21, 3:13]
        tmp = tmp[np.any(tmp == 0, axis=1)]
        a = 21 - tmp.shape[0]

        if a > 0:
            self.chain += 1
            if self.chain >= 3:
                self.rate *= 1.1
            if a == 4:
                self.score += self.rate * 80
            else:
                self.score += self.rate * a * 10
        else:
            self.chain = self.init_chain()
            self.rate = self.init_rate()
        zero = np.zeros((a, 10))
        tmp = np.concatenate([zero, tmp])
        self.board[:21, 3:13] = tmp

    def check_dead(self):
        return np.any(self.board[1, 3:13] > 0)

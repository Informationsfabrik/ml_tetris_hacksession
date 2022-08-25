from enum import IntEnum

import numpy as np


class MinoType(IntEnum):
    I = 0
    L = 1
    J = 2
    Z = 3
    S = 4
    T = 5
    O = 6


class Mino:
    """A "piece" on the board of the game.
    Each mino has a type that determines its shape.
    It has a x and y coordinate.
    Each mino covers 4 cells in various shapes.
    """

    MINOS = [
        [[0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]],  # line / I piece
        [[3, 0, 0], [3, 3, 3], [0, 0, 0]],  # L piece
        [[0, 0, 4], [4, 4, 4], [0, 0, 0]],  # J piece
        [[0, 5, 5], [5, 5, 0], [0, 0, 0]],  # Z piece
        [[6, 6, 0], [0, 6, 6], [0, 0, 0]],  # S piece
        [[0, 7, 0], [7, 7, 7], [0, 0, 0]],  # T piece
        [[8, 8], [8, 8]],  # square / O piece
    ]

    def __init__(self, x: int, y: int, type_: MinoType):
        self.x = x
        self.y = y
        self.type_ = type_
        self.mino = np.array(self.MINOS[type_.value])

    def collision(self, board: np.ndarray) -> bool:
        """Return true if this mino overlays with the board in at least one cell.

        Args:
            board (np.ndarray): the current board

        Returns:
            bool: true if a collision with the board
        """
        h, w = self.mino.shape
        return np.sum(board[self.y : self.y + h, self.x : self.x + w] * self.mino) > 0

    def copy(self):
        """Create a copy of this mino

        Returns:
            Mino: the copy
        """
        new_mino = Mino(self.x, self.y, self.type_)
        new_mino.mino = np.copy(self.mino)
        return new_mino

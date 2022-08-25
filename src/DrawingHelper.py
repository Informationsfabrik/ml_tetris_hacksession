import itertools

import cv2
import numpy as np

W = 20
H = 20
COLORS = [
    [0, 0, 0],
    [128, 128, 128],
    [0, 191, 255],
    [65, 105, 225],
    [255, 165, 0],
    [0, 255, 0],
    [220, 20, 60],
    [186, 85, 211],
    [255, 255, 0],
]


def draw(board, minos, score):
    h, w = board.shape
    canvas = np.zeros((H * h, W * w + 100, 3))
    for y in range(h):
        for x in range(w):
            cv2.rectangle(
                canvas,
                (x * W, y * H),
                ((x + 1) * W, (y + 1) * H),
                COLORS[int(board[y][x])],
                thickness=-1,
            )
            cv2.rectangle(
                canvas,
                (x * W, y * H),
                ((x + 1) * W, (y + 1) * H),
                (100, 100, 100),
                thickness=1,
            )

    _draw_mino(canvas, minos[0])
    _draw_next(canvas, minos[1:], (W * w + 10, 10))
    _draw_score(canvas, score, (W * w + 10, 360))
    cv2.line(canvas, (0, H * 2), (W * w, H * 2), color=(255, 0, 0), thickness=2)
    canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    # cv2.imshow("tetris", canvas)
    return canvas


def _draw_mino(canvas, mino):
    x, y = mino.x - 2, mino.y
    h, w = mino.mino.shape
    for tmpy in range(h):
        for tmpx in range(w):
            if mino.mino[tmpy][tmpx]:
                c = COLORS[mino.type_ + 2]
                cv2.rectangle(
                    canvas,
                    ((x + tmpx) * W, (y + tmpy) * H),
                    ((x + tmpx + 1) * W, (y + tmpy + 1) * H),
                    color=c,
                    thickness=-1,
                )
                cv2.rectangle(
                    canvas,
                    ((x + tmpx) * W, (y + tmpy) * H),
                    ((x + tmpx + 1) * W, (y + tmpy + 1) * H),
                    (100, 100, 100),
                    thickness=1,
                )


def _draw_next(canvas, minos, top_left_xy):
    sW, sH = 10, 10
    x, y = top_left_xy
    cv2.rectangle(canvas, (x, y), (x + 80, y + 320), (255, 255, 255), thickness=1)
    for i, mino in enumerate(minos):
        h, w = mino.mino.shape
        offsetx = int(40 - sW * (w / 2))
        offsety = int(40 - sH * (h / 2))
        for tmpy, tmpx in itertools.product(range(h), range(w)):
            if mino.mino[tmpy][tmpx]:
                c = COLORS[mino.type_ + 2]
                cv2.rectangle(
                    canvas,
                    (x + offsetx + tmpx * sW, y + offsety + i * 80 + tmpy * sH),
                    (
                        x + offsetx + (tmpx + 1) * sW,
                        y + offsety + i * 80 + (tmpy + 1) * sH,
                    ),
                    color=c,
                    thickness=-1,
                )


def _draw_score(canvas, score, p):
    cv2.putText(
        canvas,
        str(score),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        thickness=2,
    )

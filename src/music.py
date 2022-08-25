import os
import random
import wave

import pyaudio

from config import MUSIC_BASE_PATH


def init_songs():

    songs = list(os.walk(MUSIC_BASE_PATH))[0][2]
    songs = [MUSIC_BASE_PATH + song for song in songs]
    return songs


songs = init_songs()


def play():
    chunk = 1024
    with wave.open(random.choice(songs)) as wf:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        data = wf.readframes(chunk)
        while len(data):
            stream.write(data)
            data = wf.readframes(chunk)
        stream.stop_stream()
    p.terminate()

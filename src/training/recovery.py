"""Atomic checkpoints and cooperative interruption at optimizer-step boundaries."""
from contextlib import contextmanager
import os
import random
import signal
import threading

import numpy as np
import torch


class TrainingInterrupted(Exception):
    """Training stopped after saving a consistent recovery checkpoint."""


def atomic_save(payload, path):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def rng_state():
    numpy = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=[numpy[0], numpy[1].tolist(), *numpy[2:]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def interruption_flag():
    """Ctrl+C/SIGTERM requests a save; do not interrupt an optimizer update."""
    requested = threading.Event()
    previous = {}
    def request(signum, frame):
        requested.set()
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, request)
        yield requested.is_set
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

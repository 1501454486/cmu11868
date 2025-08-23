import sys
from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast
from queue import Queue
from threading import Thread

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda


InQueue = Queue
OutQueue = Queue

from contextlib import contextmanager
from typing import Generator, List, Union, cast
@contextmanager
def use_device(device: torch.device) -> Generator[None, None, None]:
    """:func:`torch.cuda.device` for either CPU or CUDA device."""
    if device.type != "cuda":
        yield
        return

    with torch.cuda.device(device):
        yield

class Task:
    """Task is a wrapper around a compute function that can be executed on a worker thread.
    """
    def __init__(
        self, compute) -> None:
        self._compute = compute
        self._grad_enabled = torch.is_grad_enabled()

    def compute(self):
        with torch.set_grad_enabled(self._grad_enabled):
            return self._compute()


def worker(in_queue: InQueue, out_queue: OutQueue, device: torch.device) -> None:
    """Main loop of a worker thread.
    
    The worker thread takes tasks from the input queue, computes them, and puts the results in the output queue.
    """
    with use_device(device):
        while True:
            task = in_queue.get()

            if task is None:
                break

            try:
                batch = task.compute()
            except Exception:
                exc_info = sys.exc_info()
                out_queue.put((False, exc_info))
                continue

            out_queue.put((True, (task, batch)))

    done = (False, None)
    out_queue.put(done)


def create_workers(devices: List[torch.device],) -> Tuple[List[InQueue], List[OutQueue]]:
    """Spawns worker threads. A worker thread is bound to a device.
    
    For each device, a pair of queues is created: one for input tasks and one for output results.
    To submit a task to a worker, put it in the corresponding input queue.
    To get the result of a task, get it from the corresponding output queue.
    """
    in_queues: List[InQueue] = []
    out_queues: List[OutQueue] = []

    # Spawn workers.
    workers: Dict[torch.device, Tuple[InQueue, OutQueue]] = {}

    def normalize_device(device: torch.device) -> torch.device:
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", index=torch.cuda.current_device())

        if device.type == "cpu" and device.index is not None:
            return torch.device("cpu")

        return device

    for device in devices:
        device = normalize_device(device)

        try:
            in_queue, out_queue = workers[device]
        except KeyError:
            in_queue = Queue()
            out_queue = Queue()
            workers[device] = (in_queue, out_queue)

            t = Thread(target=worker, args=(in_queue, out_queue, device), daemon=True,)
            t.start()

        in_queues.append(in_queue)
        out_queues.append(out_queue)

    return (in_queues, out_queues)

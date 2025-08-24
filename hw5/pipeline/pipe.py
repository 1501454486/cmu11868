from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

from matplotlib.bezier import split_bezier_intersecting_with_closedpath
import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    current_batch = -1
    for timestep in range(num_batches + num_partitions - 1):
        schedule = []
        if current_batch < num_batches - 1:
            current_batch += 1
        for partition in range(num_partitions)[::-1]:
            if partition > timestep:
                continue
            if current_batch + partition < timestep:
                break
            schedule.append((timestep - partition, partition))

        yield schedule
    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2
        # Split input using self.split_size
        batches = list(torch.split(x, self.split_size, dim = 0))
        num_batches = len(batches)
        for schedule in _clock_cycles(num_batches, len(self.devices)):
            self.compute(batches, schedule)
        # NOTE: keep results in the same device(last)!
        result = torch.cat([b.to(self.devices[-1]) for b in batches], dim = 0)
        return result
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        # traverse schedule
        for microbatch_idx, partition_idx in schedule:
            input_tensor = batches[microbatch_idx]
            partition = partitions[partition_idx]
            device = devices[partition_idx]
            
            # use lambda function to define compute function
            compute_func = lambda data=input_tensor, model=partition, dev=device: model(data.to(dev))
            # Create task
            task = Task(compute = compute_func)
            # Submit task to corresponding (partition_idx) partition in_queue
            self.in_queues[partition_idx].put(task)

        # traverse again, and put results back to batches
        for microbatch_idx, partition_idx in schedule:
            success, payload = self.out_queues[partition_idx].get()
            if success is True:
                task, batch_output = payload
                batches[microbatch_idx] = batch_output
            else:
                exc_info = payload
                raise exc_info[1].with_traceback(exc_info[2])
        # END ASSIGN5_2_2


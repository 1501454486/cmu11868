from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


class Partition():
    """
    Used to define a dataset class to return the data according to partitioned indices
    """
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN ASSIGN5_1_1
        return self.data[self.index[index]]
        # END ASSIGN5_1_1

class DataPartitioner():
    """
    Used to partition any datasets according to different workload defined as `sizes`
    """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN ASSIGN5_1_1
        data_size = len(data)
        indices = list(range(data_size))
        rng.shuffle(indices)
        pointer = 0
        for size in sizes:
            count = int(data_size * size)
            partition_indices = indices[pointer : pointer + count]
            self.partitions.append(partition_indices)
            pointer += count
        # END ASSIGN5_1_1

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN ASSIGN5_1_1
        index = self.partitions[partition]
        return Partition(self.data, index)
        # END ASSIGN5_1_1

def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN ASSIGN5_1
    partition = [1 / world_size] * world_size
    data_partitioner = DataPartitioner(dataset, sizes = partition)
    partitioned_dataset = data_partitioner.use(rank)
    return DataLoader(
        dataset = partitioned_dataset,
        batch_size = batch_size,
        collate_fn = collate_fn
    )
    # END ASSIGN5_1

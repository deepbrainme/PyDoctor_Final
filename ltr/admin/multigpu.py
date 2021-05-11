import os

import torch.nn as nn
from gpustat import GPUStatCollection


def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.DataParallel))


class MultiGPU(nn.DataParallel):
    """Wraps a network to allow simple multi-GPU training."""
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)


def query_gpu():
    gpu_stat = GPUStatCollection.new_query()
    gpu_free_idx = 0 if gpu_stat[0].memory_free >= gpu_stat[1].memory_free else 1
    print('Query time: {} -- GPU[{}]: {}MB -- '.format(gpu_stat.query_time, gpu_free_idx,
                                                       gpu_stat[gpu_free_idx].memory_free))
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpu_free_idx)
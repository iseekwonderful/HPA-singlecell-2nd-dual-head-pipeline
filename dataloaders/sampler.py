import random
from torch.utils.data import Dataset, Sampler
import numpy as np


class RandomBatchSampler(Sampler):
    def __init__(self, df, batch_size, cfg, miu):
        super().__init__(self)
        self.df = df.reset_index()
        self.batch_size = batch_size
        self.cfg = cfg
        self.miu = miu
        print('[W] Using Batch Batch sampler ^_^ with init μ={}'.format(miu))
        if self.batch_size not in [64, 128, 256]:
            raise Exception('Not acceptable batch_size')

    def __iter__(self):
        for x in range(len(self.df) // self.batch_size):
            cnt = round(np.random.normal(self.miu, 0.5))
            pos = self.df[self.df.target == 1].sample(cnt).index
            neg = self.df[self.df.target == 0].sample(self.batch_size - cnt).index
            # index = list(self.df[self.df.experiment == random.sample(self.targets, 1)[0]].sample(16).index)
            # print(pos)
            yield list(pos) + list(neg)

    def __len__(self):
        return len(self.df) // self.batch_size

    def update_miu(self, miu):
        self.miu = miu

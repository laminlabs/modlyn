import asyncio
from itertools import islice

import numpy as np
import zarr
import zarr.core.sync as zsync


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class ZarrArrayDataset:
    def __init__(
        self, array: zarr.Array, shuffle: bool = True, preload_nchunks: int = 3
    ):
        self.array = array
        self.shuffle = shuffle
        self.preload_chunks = preload_nchunks

        self.n_obs = self.array.shape[0]

        self.chunk = self.array.chunks[0]
        self.chunks = np.arange(self.array.nchunks)

    def chunk_slice(self, i: int):
        return slice(self.chunk * i, min(self.chunk * (i + 1), self.n_obs))

    def __iter__(self):
        chunks = np.random.permutation(self.chunks) if self.shuffle else self.chunks  # noqa: NPY002

        for batch in batched(chunks, self.preload_chunks):
            for chunk in zsync.sync(self.fetch_chunks(batch)):
                yield np.random.permutation(chunk) if self.shuffle else chunk  # noqa: NPY002

    async def fetch_chunks(self, chunk_idxs: list[int]):
        tasks = [
            self.array._async_array.getitem(self.chunk_slice(i)) for i in chunk_idxs
        ]
        return await asyncio.gather(*tasks)

    def __len__(self):
        return self.n_obs

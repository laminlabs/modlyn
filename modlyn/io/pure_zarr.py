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
        self, arrays: list[zarr.Array], shuffle: bool = True, preload_nchunks: int = 3
    ):
        self.arrays = arrays
        self.shuffle = shuffle
        self.preload_chunks = preload_nchunks

        self.n_obs_list: list[int] = []  # number of observations for each array
        self.chunk_lengths: list[int] = []  # chunk length for each array
        arrays_chunks: list[list[int]] = []  # list of chunk indices for each array
        arrays_nchunks: list[int] = []  # number of chunks for each array
        for array in arrays:
            self.n_obs_list.append(array.shape[0])
            self.chunk_lengths.append(array.chunk[0])
            array_nchunks = array.nchunks
            arrays_nchunks.append(array_nchunks)
            arrays_chunks.append(np.arange(array_nchunks))

        self.n_obs = sum(self.n_obs_list)
        self.chunks = np.hstack(arrays_chunks)
        self.array_idxs = np.repeat(np.arange(len(self.arrays)), arrays_nchunks)
        # pre-compute chunk slices
        self.chunk_slices: list[slice] = []
        for i, chunk_idx in enumerate(self.chunks):
            self.chunk_slices.append(self._chunk_slice(chunk_idx, self.array_idxs[i]))

    def _chunk_slice(self, chunk_idx: int, array_idx: int):
        chunk_length = self.chunk_lengths[array_idx]
        array_n_obs = self.n_obs_list[array_idx]

        start = chunk_length * chunk_idx
        stop = min(chunk_length * (chunk_idx + 1), array_n_obs)
        return slice(start, stop)

    async def fetch_chunks(self, chunk_idxs: list[int]):
        tasks = []
        for i in chunk_idxs:
            array_idx = self.array_idxs[i]
            array = self.arrays[array_idx]
            tasks.append(array._async_array.getitem(self.chunk_slices[i]))
        if len(tasks) == 1:
            return tasks[0]
        else:
            return await asyncio.gather(*tasks)

    def __iter__(self):
        chunks = np.arange(len(self.chunks))
        if self.shuffle:
            chunks = np.random.permutation(chunks)  # noqa: NPY002

        for batch in batched(chunks, self.preload_chunks):
            for chunk_arr in zsync.sync(self.fetch_chunks(batch)):
                yield np.random.permutation(chunk_arr) if self.shuffle else chunk_arr  # noqa: NPY002

    def __len__(self):
        return self.n_obs

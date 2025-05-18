import asyncio
from itertools import islice
from os import PathLike

import numpy as np
import zarr
import zarr.core.sync as zsync
from upath import UPath
from zarr.core.array import BasicIndexer, default_buffer_prototype


def shards_dir_to_arrays(path: PathLike) -> list[zarr.Array]:
    upath = UPath(path)
    arrays = []
    for p in upath.iterdir():
        if p.suffix != ".zarr":
            continue
        p_x = p / "X"
        if p_x.protocol == "":
            store = p_x.as_posix()
        else:
            store = zarr.storage.FsspecStore.from_upath(UPath(p_x, asynchronous=True))
        arrays.append(zarr.open(store, mode="r"))
    return arrays


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class ZarrArraysDataset:
    def __init__(
        self,
        arrays: list[zarr.Array],
        shuffle: bool = True,
        preload_nchunks: int = 8,
        concat_preloaded: bool = True,
    ):
        self.arrays = arrays
        self.shuffle = shuffle
        self.preload_chunks = preload_nchunks
        self.concat_preloaded = concat_preloaded

        self.n_obs_list: list[int] = []  # number of observations for each array
        self.chunks_lengths: list[int] = []  # chunk length for each array
        arrays_chunks: list[list[int]] = []  # list of chunk indices for each array
        arrays_nchunks: list[int] = []  # number of chunks for each array
        for array in arrays:
            self.n_obs_list.append(array.shape[0])
            self.chunks_lengths.append(array.chunks[0])
            array_nchunks = array.nchunks
            arrays_nchunks.append(array_nchunks)
            arrays_chunks.append(np.arange(array_nchunks))

        self.n_obs = sum(self.n_obs_list)
        # assumes the same for all arrays
        array0 = arrays[0]
        self.n_vars = array0.shape[1]
        self.dtype = array0.dtype
        self.order = array0.order

        self.chunks = np.hstack(arrays_chunks)
        self.array_idxs = np.repeat(np.arange(len(self.arrays)), arrays_nchunks)
        # pre-compute chunk slices
        # slices are needed because we want to iterate over (logical) chunks, not (physical) shards
        # but in zarr array.blocks[i] returns whole shards unlike dask
        self.chunks_slices: list[slice] = []
        for i, chunk_idx in enumerate(self.chunks):
            self.chunks_slices.append(self._chunk_slice(chunk_idx, self.array_idxs[i]))

    def _chunk_slice(self, chunk_idx: int, array_idx: int):
        chunk_length = self.chunks_lengths[array_idx]
        array_n_obs = self.n_obs_list[array_idx]

        start = chunk_length * chunk_idx
        stop = min(chunk_length * (chunk_idx + 1), array_n_obs)
        return slice(start, stop)

    # fast zero-copy concatenated chunks
    async def fetch_chunks_concat(self, chunk_idxs: list[int]):
        prototype = default_buffer_prototype()

        local_slices = []
        out_n_obs = 0
        for idx in chunk_idxs:
            chunk_slice = self.chunks_slices[idx]
            chunk_length = chunk_slice.stop - chunk_slice.start
            start = out_n_obs
            out_n_obs += chunk_length
            local_slices.append(slice(start, out_n_obs))

        out = np.empty((out_n_obs, self.n_vars), dtype=self.dtype, order=self.order)
        tasks = []
        for i, idx in enumerate(chunk_idxs):
            chunk_slice = self.chunks_slices[idx]
            array_idx = self.array_idxs[idx]
            array = self.arrays[array_idx]
            indexer = BasicIndexer(
                chunk_slice,
                shape=array.metadata.shape,
                chunk_grid=array.metadata.chunk_grid,
            )
            # this can also be a gpu buffer
            buffer = prototype.nd_buffer.from_numpy_array(out[local_slices[i]])
            task = array._async_array._get_selection(
                indexer, prototype=prototype, out=buffer
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

        return out

    async def fetch_chunks(self, chunk_idxs: list[int]):
        tasks = []
        for i in chunk_idxs:
            array_idx = self.array_idxs[i]
            array = self.arrays[array_idx]
            tasks.append(array._async_array.getitem(self.chunks_slices[i]))
        if len(tasks) == 1:
            return await tasks[0]
        else:
            return await asyncio.gather(*tasks)

    def __iter__(self):
        chunks_global = np.arange(len(self.chunks))
        if self.shuffle:
            np.random.shuffle(chunks_global)  # noqa: NPY002

        for batch in batched(chunks_global, self.preload_chunks):
            # concat chunk arrays in the batch
            if len(batch) > 1 and self.concat_preloaded:
                batch_arr = zsync.sync(self.fetch_chunks_concat(batch))
                if self.shuffle:
                    # important
                    # use shuffle to avoid making a copy
                    np.random.shuffle(batch_arr)  # noqa: NPY002
                yield batch_arr
            else:
                # just return chunk arrays from the batch one by one
                for chunk_arr in zsync.sync(self.fetch_chunks(batch)):
                    if self.shuffle:
                        # use shuffle to avoid making a copy
                        np.random.shuffle(chunk_arr)  # noqa: NPY002
                    yield chunk_arr

    def __len__(self):
        return self.n_obs

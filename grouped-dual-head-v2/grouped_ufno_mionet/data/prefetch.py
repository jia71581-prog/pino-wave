"""Bounded CPU prefetching helpers.

The queue is deliberately bounded: a slow GPU cannot cause an unbounded pile
of decoded wavefields in host memory.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from torch.utils.data import DataLoader


def identity_collate(batch):
    return batch


def bounded_prefetch(source: Iterable, max_prefetch: int = 2) -> Iterator:
    if max_prefetch <= 0:
        raise ValueError("max_prefetch must be positive")
    iterator = iter(source)
    with ThreadPoolExecutor(max_workers=max_prefetch, thread_name_prefix="grouped-prefetch") as pool:
        queue: deque[Future] = deque()
        exhausted = False
        while queue or not exhausted:
            while not exhausted and len(queue) < max_prefetch:
                try: item = next(iterator)
                except StopIteration: exhausted = True; break
                queue.append(pool.submit(lambda value: value, item))
            if queue:
                yield queue.popleft().result()


class BoundedPrefetchLoader:
    def __init__(self, loader: Iterable, prefetch_batches: int = 2):
        if prefetch_batches <= 0: raise ValueError("prefetch_batches must be positive")
        self.loader, self.prefetch_batches = loader, int(prefetch_batches)

    def __iter__(self):
        return bounded_prefetch(self.loader, self.prefetch_batches)

    def __len__(self):
        return len(self.loader)


def make_dataloader(dataset, *, batch_size: int | None = 24, batch_sampler=None,
                    num_workers: int = 0, prefetch_batches: int = 2,
                    shuffle: bool = False, collate_fn=None, **kwargs):
    """Create a DataLoader with bounded worker prefetch and safe defaults."""
    # GroupedSample is intentionally not a default-collatable tensor tuple;
    # leave records as a list unless the caller supplies ``pack_groups``.
    if collate_fn is None:
        collate_fn = identity_collate
    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                            collate_fn=collate_fn, **kwargs)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, collate_fn=collate_fn, **kwargs)
    return BoundedPrefetchLoader(loader, prefetch_batches)


# Friendly aliases used by training scripts written against earlier drafts.
PrefetchLoader = BoundedPrefetchLoader
prefetch_dataloader = make_dataloader

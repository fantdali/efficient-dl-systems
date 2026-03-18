import bisect
import random
from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler


MAX_LENGTH = 640


class BrainDataset(Dataset):
    """Pad every sample to a fixed max_length (BRAIN approach)."""

    def __init__(self, texts: list, tokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.input_ids: list[list[int]] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            self.input_ids.append(ids[:max_length])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        seq = self.input_ids[idx]
        padded = seq + [0] * (self.max_length - len(seq))
        return torch.tensor(padded, dtype=torch.long)


class BigBrainDataset(Dataset):
    """Pad only inside collate_fn up to the longest sample in the batch (BIG BRAIN)."""

    def __init__(self, texts: list, tokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.input_ids: list[list[int]] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            self.input_ids.append(ids[:max_length])

    def __getitem__(self, idx: int):
        seq = self.input_ids[idx]
        return seq, len(seq)

    def __len__(self):
        return len(self.input_ids)

    @staticmethod
    def collate_fn(batch):
        seqs, lengths = zip(*batch)
        max_batch_len = max(lengths)
        padded = torch.zeros(len(seqs), max_batch_len, dtype=torch.long)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded


class UltraBigBrainDataset(Dataset):
    """
    Group sequences by length - pad only to max length in each batch.
    Stores a hash table length - list[index] for O(1) bucket lookup.
    """

    def __init__(self, texts: list, tokenizer, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.input_ids: list[list[int]] = []
        # hash table: sequence length - list of dataset indices
        self.length_to_indices: dict[int, list[int]] = {}

        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            ids = ids[:max_length]
            idx = len(self.input_ids)
            length = len(ids)
            self.input_ids.append(ids)
            if length not in self.length_to_indices:
                self.length_to_indices[length] = []
            self.length_to_indices[length].append(idx)

        self.sorted_lengths: list[int] = sorted(self.length_to_indices.keys())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        seq = self.input_ids[idx]
        return seq, len(seq)

    @staticmethod
    def collate_fn(batch):
        seqs, lengths = zip(*batch)
        max_batch_len = max(lengths)
        padded = torch.zeros(len(seqs), max_batch_len, dtype=torch.long)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded


class UltraBigBrainBatchSampler(Sampler):
    """
    Yields batches where the max length spread is at most k tokens.
    __init__: O(n) - builds sorted flat index arrays.
    Each yield from __iter__: O(batch_size) - random sample from a bisect range.
    """

    def __init__(self, dataset: UltraBigBrainDataset, batch_size: int, k: int = 1):
        self.batch_size = batch_size
        self.k = k

        # Build flat array sorted by length O(n)
        self.sorted_indices: list[int] = []
        self.sorted_item_lengths: list[int] = []
        for length in dataset.sorted_lengths:
            for idx in dataset.length_to_indices[length]:
                self.sorted_indices.append(idx)
                self.sorted_item_lengths.append(length)

        self.n = len(self.sorted_indices)

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        # Shuffle within each length group - O(n) once per epoch
        shuffled: list[int] = []
        i = 0
        while i < self.n:
            j = i
            cur_len = self.sorted_item_lengths[i]
            while j < self.n and self.sorted_item_lengths[j] == cur_len:
                j += 1
            group = list(self.sorted_indices[i:j])
            random.shuffle(group)
            shuffled.extend(group)
            i = j

        for _ in range(len(self)):
            # Pick a random position and read its length
            pos = random.randrange(self.n)
            pivot_len = self.sorted_item_lengths[pos]

            # Binary search for the k-spread range - O(log max_length) ~ O(1)
            lo = bisect.bisect_left(self.sorted_item_lengths, pivot_len - self.k)
            hi = bisect.bisect_right(self.sorted_item_lengths, pivot_len + self.k)

            # Sample batch_size indices from [lo, hi) - O(batch_size)
            actual_size = min(self.batch_size, hi - lo)
            sampled_positions = random.sample(range(lo, hi), actual_size)
            yield [shuffled[p] for p in sampled_positions]


# ---------------------------------------------------------------------------
# Segment tree used by OBFD packing
# ---------------------------------------------------------------------------

class _SegTree:
    """Segment tree over remaining bin capacities [0..max_cap].

    Supports finding the best-fit bin (minimum remaining capacity >= length)
    in O(log max_cap) time.
    """

    def __init__(self, max_cap: int):
        self.n = max_cap + 1
        self.INF = max_cap + 1
        # tree[node] = minimum capacity value present in that node's range
        # INF if the range contains no bins
        self.tree = [self.INF] * (4 * self.n)
        # for each capacity value, which bin indices have that remaining capacity
        self.cap_to_bins: list[list[int]] = [[] for _ in range(self.n)]

    def _update(self, node: int, start: int, end: int, pos: int) -> None:
        if start == end:
            self.tree[node] = pos if self.cap_to_bins[pos] else self.INF
            return
        mid = (start + end) // 2
        if pos <= mid:
            self._update(2 * node, start, mid, pos)
        else:
            self._update(2 * node + 1, mid + 1, end, pos)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def _query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        if r < start or end < l:
            return self.INF
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return min(
            self._query(2 * node, start, mid, l, r),
            self._query(2 * node + 1, mid + 1, end, l, r),
        )

    def add_bin(self, capacity: int, bin_idx: int) -> None:
        self.cap_to_bins[capacity].append(bin_idx)
        self._update(1, 0, self.n - 1, capacity)

    def remove_bin(self, capacity: int, bin_idx: int) -> None:
        self.cap_to_bins[capacity].remove(bin_idx)
        self._update(1, 0, self.n - 1, capacity)

    def best_fit(self, length: int) -> tuple[int, int]:
        """Return (capacity, bin_idx) of best-fit bin, or (-1, -1) if none."""
        cap = self._query(1, 0, self.n - 1, length, self.n - 1)
        if cap >= self.INF:
            return -1, -1
        return cap, self.cap_to_bins[cap][-1]


class UltraDuperBigBrainDataset(Dataset):
    """Pack sequences into bins of max_length (ULTRA DUPER BIG BRAIN).

    packing : 'basic'  — greedy sequential packing
              'ffd'    — First-Fit Decreasing (O(N*M))
              'obfd'   — Optimised Best-Fit Decreasing via segment tree (O(N log L))

    Each item is a (packed_ids [L], attn_mask [L, L]) pair.
    attn_mask is an additive float mask (0 = attend, -inf = block) that prevents
    cross-contamination between packed sequences and attends causally within each
    sequence.
    """

    def __init__(self, texts: list, tokenizer, max_length: int = MAX_LENGTH,
                 packing: str = 'basic'):
        self.max_length = max_length

        # Tokenise once; skip empty and over-length sequences
        all_ids: list[list[int]] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids and len(ids) <= max_length:
                all_ids.append(ids)

        # bins[i] = (flat token list, [len_seq0, len_seq1, ...])
        self.bins: list[tuple[list[int], list[int]]] = []

        if packing == 'basic':
            self._basic_packing(all_ids)
        elif packing == 'ffd':
            self._ffd_packing(all_ids)
        elif packing == 'obfd':
            self._obfd_packing(all_ids)
        else:
            raise ValueError(f"Unknown packing mode: {packing!r}")

    # ------------------------------------------------------------------
    # Packing algorithms
    # ------------------------------------------------------------------

    def _basic_packing(self, all_ids: list[list[int]]) -> None:
        """Greedy packing in corpus order."""
        cur_ids: list[int] = []
        cur_seqlens: list[int] = []
        cur_len = 0

        for ids in all_ids:
            l = len(ids)
            if cur_len + l > self.max_length:
                if cur_ids:
                    self.bins.append((cur_ids, cur_seqlens))
                cur_ids, cur_seqlens, cur_len = list(ids), [l], l
            else:
                cur_ids.extend(ids)
                cur_seqlens.append(l)
                cur_len += l

        if cur_ids:
            self.bins.append((cur_ids, cur_seqlens))

    def _ffd_packing(self, all_ids: list[list[int]]) -> None:
        """First-Fit Decreasing — O(N * M)."""
        sorted_ids = sorted(all_ids, key=len, reverse=True)
        bin_ids: list[list[int]] = []
        bin_seqlens: list[list[int]] = []
        bin_remaining: list[int] = []

        for ids in sorted_ids:
            l = len(ids)
            placed = False
            for i in range(len(bin_ids)):
                if bin_remaining[i] >= l:
                    bin_ids[i].extend(ids)
                    bin_seqlens[i].append(l)
                    bin_remaining[i] -= l
                    placed = True
                    break
            if not placed:
                bin_ids.append(list(ids))
                bin_seqlens.append([l])
                bin_remaining.append(self.max_length - l)

        self.bins = list(zip(bin_ids, bin_seqlens))

    def _obfd_packing(self, all_ids: list[list[int]]) -> None:
        """Optimised Best-Fit Decreasing via segment tree — O(N log L)."""
        sorted_ids = sorted(all_ids, key=len, reverse=True)
        seg = _SegTree(self.max_length)
        bin_ids: list[list[int]] = []
        bin_seqlens: list[list[int]] = []
        bin_remaining: list[int] = []

        for ids in sorted_ids:
            l = len(ids)
            cap, bin_idx = seg.best_fit(l)

            if bin_idx == -1:
                # Open a new bin
                new_idx = len(bin_ids)
                bin_ids.append(list(ids))
                bin_seqlens.append([l])
                remaining = self.max_length - l
                bin_remaining.append(remaining)
                if remaining > 0:
                    seg.add_bin(remaining, new_idx)
            else:
                # Place in best-fit bin
                seg.remove_bin(cap, bin_idx)
                bin_ids[bin_idx].extend(ids)
                bin_seqlens[bin_idx].append(l)
                new_remaining = cap - l
                bin_remaining[bin_idx] = new_remaining
                if new_remaining > 0:
                    seg.add_bin(new_remaining, bin_idx)

        self.bins = list(zip(bin_ids, bin_seqlens))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, seq_lens = self.bins[idx]

        # Pad token ids to max_length
        padded = ids + [0] * (self.max_length - len(ids))
        packed_ids = torch.tensor(padded, dtype=torch.long)

        # Build block-causal attention mask — vectorised
        # seq_idx[pos] = 1-based sequence index (0 = padding)
        seq_idx = torch.zeros(self.max_length, dtype=torch.long)
        pos = 0
        for s_idx, sl in enumerate(seq_lens, start=1):
            seq_idx[pos:pos + sl] = s_idx
            pos += sl

        # can_attend[i, j] = True if position i may attend to position j
        same_seq = (seq_idx.unsqueeze(1) == seq_idx.unsqueeze(0))  # [L, L]
        non_pad  = (seq_idx.unsqueeze(1) > 0) & (seq_idx.unsqueeze(0) > 0)
        causal   = torch.ones(self.max_length, self.max_length, dtype=torch.bool).tril()
        can_attend = same_seq & non_pad & causal

        # Additive float mask: 0 = attend, -inf = block
        attn_mask = torch.zeros(self.max_length, self.max_length)
        attn_mask.masked_fill_(~can_attend, float('-inf'))

        return packed_ids, attn_mask

    @staticmethod
    def collate_fn(batch):
        ids_list, mask_list = zip(*batch)
        return torch.stack(ids_list), torch.stack(mask_list)

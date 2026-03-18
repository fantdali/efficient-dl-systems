from enum import Enum

from statistics import mean, median
import torch
import transformer
from tqdm.auto import tqdm


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_gpt2_model(tokenizer) -> torch.nn.Module:
    model = transformer.TransformerModel(
        ntoken=tokenizer.vocab_size,
        d_model=1024,
        nhead=8,
        d_hid=1024,
        nlayers=1,
        dropout=0.1,
    )
    return model


def run_epoch(dataloader, tokenizer, device) -> None:
    warmup_batches = 5

    model = get_gpt2_model(tokenizer).to(device)
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    times = []
    batch_sizes = []
    sequence_lengths = []

    i = 0
    perf_batches = 5000
    nheads = 8  # matches get_gpt2_model config
    # forward pass only
    for batch in tqdm(dataloader):
        if warmup_batches > 0:
            is_warmup = True
            warmup_batches -= 1
        else:
            is_warmup = False

        # UltraDuperBigBrainDataset returns (packed_ids, attn_mask)
        if isinstance(batch, (tuple, list)):
            packed_ids, attn_masks = batch   # [B, L], [B, L, L]
        else:
            packed_ids = batch               # [B, L]
            attn_masks = None

        batch_size = packed_ids.size(0)
        seq_len = packed_ids.size(1)

        batch_sizes.append(batch_size)
        sequence_lengths.append(seq_len)

        if not is_warmup:
            start_time.record()

        packed_ids = packed_ids.to(device)
        inputs = packed_ids.t()  # [L, B]

        if attn_masks is not None:
            # Expand per-sample mask for multi-head attention: [B*nheads, L, L]
            attn_masks = attn_masks.to(device)
            src_mask = (attn_masks
                        .unsqueeze(1)
                        .expand(-1, nheads, -1, -1)
                        .reshape(batch_size * nheads, seq_len, seq_len))
        else:
            src_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        outputs = model(inputs, src_mask)  # [L, B, ntoken]
        
        if not is_warmup:
            end_time.record()
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))

        if i >= perf_batches:
            break
        i += 1

    print()
    mean_time = mean(times) if times else 0
    print(f"Mean time per batch: {mean_time:.2f} ms")
    min_time = min(times) if times else 0
    print(f"Min time per batch: {min_time:.2f} ms")
    max_time = max(times) if times else 0
    print(f"Max time per batch: {max_time:.2f} ms")
    median_time = median(times) if times else 0
    print(f"Median time per batch: {median_time:.2f} ms")

    print() 
    mean_batch_size = mean(batch_sizes) if batch_sizes else 0
    print(f"Mean batch size: {mean_batch_size:.2f}")
    max_batch_size = max(batch_sizes) if batch_sizes else 0
    print(f"Max batch size: {max_batch_size}")
    min_batch_size = min(batch_sizes) if batch_sizes else 0
    print(f"Min batch size: {min_batch_size}")
    median_batch_size = median(batch_sizes) if batch_sizes else 0
    print(f"Median batch size: {median_batch_size}")

    print()
    mean_sequence_length = mean(sequence_lengths) if sequence_lengths else 0
    print(f"Mean sequence length: {mean_sequence_length:.2f}")
    max_sequence_length = max(sequence_lengths) if sequence_lengths else 0
    print(f"Max sequence length: {max_sequence_length}")
    min_sequence_length = min(sequence_lengths) if sequence_lengths else 0
    print(f"Min sequence length: {min_sequence_length}")
    median_sequence_length = median(sequence_lengths) if sequence_lengths else 0
    print(f"Median sequence length: {median_sequence_length}")
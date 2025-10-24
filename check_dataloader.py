# pip install dataset
import os

import sys
#__package__ = "trainer"
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Add project root so `dataset.lm_dataset` can be imported when running this file directly
#import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.lm_dataset import PretrainDataset

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def decode(tok, ids):
    try:
        return tok.decode([int(x) for x in ids], skip_special_tokens=False)
    except Exception:
        try:
            return tok.batch_decode([ids], skip_special_tokens=False)[0]
        except Exception:
            return "<could not decode>"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/ubuntu/minimind/dataset/pretrain_hq.jsonl")
    parser.add_argument("--model_dir", default="/home/ubuntu/minimind/model/")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--sample_mode", choices=["loader", "raw"], default="loader",
                        help="loader: inspect DataLoader batches; raw: sample raw jsonl lines and tokenize")
    args = parser.parse_args()

    # Make paths absolute and ensure local files are used (avoid HF hub validation)
    args.data_path = os.path.abspath(args.data_path)
    args.model_dir = os.path.abspath(args.model_dir)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"data_path not found: {args.data_path}")
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"model_dir not found: {args.model_dir}")
    print(f"Using data_path={args.data_path}")
    print(f"Using model_dir={args.model_dir}")

    # init tokenizer and dataset (local_files_only avoids HF hub lookup)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    ds = PretrainDataset(args.data_path, tokenizer, max_length=512)

    if is_main_process():
        print("Dataset length:", len(ds))

    if args.sample_mode == "loader":
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        it = iter(dl)
        for i in range(args.num_batches):
            try:
                batch = next(it)
            except StopIteration:
                print("End of DataLoader")
                break

            # Expecting (X, Y, loss_mask) as in training script
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                X = batch[0]
                Y = batch[1] if len(batch) > 1 else None
                loss_mask = batch[2] if len(batch) > 2 else None
                if is_main_process():
                    print(f"\nBatch {i} - tensors:")
                    print(" X.shape:", None if X is None else tuple(X.shape))
                    print(" Y.shape:", None if Y is None else tuple(Y.shape))
                    print(" loss_mask.shape:", None if loss_mask is None else tuple(loss_mask.shape))

                    # show stats
                    with torch.no_grad():
                        x_min = int(X.min()) if X is not None else None
                        x_max = int(X.max()) if X is not None else None
                        print(" X token id range:", x_min, "~", x_max)

                    # decode first sequence in batch
                    first = X[0].tolist() if X is not None else []
                    print(" first sequence tokens (first 64):", first[:64])
                    print(" decoded first sequence:", decode(tokenizer, first))

            else:
                if is_main_process():
                    print("Batch is not a tuple/list; repr:", repr(batch))

    else:  # raw mode: sample a few raw jsonl lines and tokenize to inspect lengths
        import json, random
        if is_main_process():
            print("\nSampling raw lines and tokenizing to show length distribution (up to 100 samples)...")
        total = 0
        lengths = []
        with open(args.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sample_n = min(len(lines), 100)
        for line in random.sample(lines, sample_n):
            total += 1
            try:
                obj = json.loads(line)
                # try common keys
                text = obj.get("text") or obj.get("content") or obj.get("instruction") or obj.get("input") or str(obj)
            except Exception:
                text = line.strip()
            toks = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(toks))

        if is_main_process():
            import statistics
            print("Sampled lines:", total)
            print("lengths - min/mean/median/max:", min(lengths), statistics.mean(lengths), statistics.median(lengths), max(lengths))
            # small histogram
            bins = [0]*10
            for L in lengths:
                idx = min(9, (L*10)//(max(lengths, default=1)+1))
                bins[idx]+=1
            print("Rough histogram (10 bins):", bins)

if __name__ == "__main__":
    main()
"""Strip the ICL generation-head weights from an aligned Stack checkpoint.

stack-embedding loads a checkpoint into ``StateICLModel`` (the encoder-only architecture, no
generation head), so the cytokine-aligned checkpoint -- which carries the extra head weights
(``query_pos_embedding`` + the ``cls`` MLP) -- fails the strict state_dict load with "Unexpected
key(s)". Those weights are unused for embedding, so drop them: the remaining encoder weights are
exactly the aligned model's representation. Base (unaligned) checkpoints need no stripping.

  python scripts/strip_ckpt_head.py --in bc_large_aligned.ckpt --out bc_large_aligned_encoder.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# The keys StateICLModel rejects (matched by suffix, so any LightningModule prefix is handled).
HEAD_SUFFIXES = (
    "query_pos_embedding",
    "cls.0.weight",
    "cls.0.bias",
    "cls.2.weight",
    "cls.2.bias",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="aligned checkpoint (.ckpt)")
    ap.add_argument("--out", required=True, help="destination encoder-only checkpoint (.ckpt)")
    args = ap.parse_args()

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    removed = [k for k in list(sd) if any(k.endswith(s) or k == s for s in HEAD_SUFFIXES)]
    if not removed:
        print("no gen-head keys found -- checkpoint may already be encoder-only; copying as-is")
    for k in removed:
        del sd[k]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt["state_dict"] = sd
    else:
        ckpt = sd
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"wrote {args.out}; removed {len(removed)} gen-head keys: {removed}")


if __name__ == "__main__":
    main()

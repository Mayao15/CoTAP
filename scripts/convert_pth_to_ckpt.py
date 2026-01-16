#!/usr/bin/env python
"""
Convert a .pth weight file to the checkpoint format expected by CoTAP.

CoTAP's `utils.load_checkpoint()` accepts:
- a raw state_dict (dict: param_name -> tensor), or
- a dict with a top-level "state_dict" key.

This script wraps a raw state_dict into {"state_dict": ...} and saves it as .ckpt.
"""

import argparse
import os
from typing import Any, Dict

import torch


def _as_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    raise ValueError(
        "Unsupported checkpoint structure. Expected a raw state_dict or a dict containing 'state_dict'."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_pth", required=True, help="Input .pth path")
    parser.add_argument("--out_ckpt", required=True, help="Output .ckpt path")
    args = parser.parse_args()

    in_pth = args.in_pth
    out_ckpt = args.out_ckpt

    obj = torch.load(in_pth, map_location="cpu")
    state_dict = _as_state_dict(obj)

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    payload = {"state_dict": state_dict}
    torch.save(payload, out_ckpt)

    print(f"[OK] Loaded: {in_pth}")
    print(f"[OK] state_dict params: {len(state_dict)}")
    print(f"[OK] Saved: {out_ckpt}")


if __name__ == "__main__":
    main()


















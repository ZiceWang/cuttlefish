# --- repo path bootstrap: flat sibling imports + runs/data/tokenizer paths ---
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[1]
for _d in [_EXP, *sorted(_p for _p in _EXP.iterdir() if _p.is_dir() and _p.name != "__pycache__")]:
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
HERE = _EXP.parent
# ---------------------------------------------------------------------------
import json

import torch
from tokenizers import Tokenizer

from train_sft import DATA_DIR, IGNORE_INDEX, PackedChatDataset, encode_messages


def test_chatml_special_tokens_and_assistant_mask():
    tokenizer = Tokenizer.from_file(str(DATA_DIR / "bpe4099_chatml.json"))
    assert tokenizer.get_vocab_size() == 4099
    assert [tokenizer.token_to_id(x) for x in ("<|im_start|>", "<|im_end|>", "<|pad|>")] == [4096, 4097, 4098]
    ids, mask = encode_messages([
        {"role": "system", "content": "Be accurate."},
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": "Answer."},
    ], tokenizer)
    supervised = tokenizer.decode([x for x, keep in zip(ids, mask) if keep],
                                  skip_special_tokens=False)
    assert supervised == "Answer.<|im_end|>"


def test_packing_preserves_short_tail(tmp_path):
    path = tmp_path / "tiny.jsonl"
    row = {"messages": [{"role": "user", "content": "Q"},
                         {"role": "assistant", "content": "A"}]}
    path.write_text(json.dumps(row) + "\n")
    dataset = PackedChatDataset(path, DATA_DIR / "bpe4099_chatml.json", 64)
    x, labels = next(iter(dataset))
    assert x.shape == labels.shape == (64,)
    assert torch.any(labels != IGNORE_INDEX)

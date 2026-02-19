"""
Standalone test for OLMo-core training data insertions.

Tests the InsertionMapReader (in olmo-core) and the insertion logic in
data_loader._get_dataset_item() without needing the full olmo_core.data
import chain (which requires torch >= 2.6 / DeviceMesh).

Two tests:
  1. Raw token insertion — verify tokens are inserted at the correct position
  2. Text encode/decode round-trip with the dolma2 tokenizer
"""

import os
import sys
import tempfile

import h5py
import numpy as np
import torch

from pretrain_experiments.insertion_map import InsertionMapWriter

# Import the OLMo-core InsertionMapReader directly from the file,
# bypassing olmo_core.data.__init__.py (which pulls in DTensor and needs torch >= 2.6).
import importlib.util

_insertion_map_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "OLMo-core", "src",
    "olmo_core", "data", "insertion_map.py"
)
_spec = importlib.util.spec_from_file_location("olmo_core_insertion_map", _insertion_map_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
InsertionMapReader = _mod.InsertionMapReader


def create_optimized_insertion_map(tmpdir, insertion_dict):
    """Create an optimized HDF5 insertion map file, return the path."""
    working_path = os.path.join(tmpdir, "insertion_map.h5")
    optimized_path = os.path.join(tmpdir, "insertion_map_optimized.h5")
    writer = InsertionMapWriter(working_path)
    writer.write_dict(insertion_dict)
    writer.save_optimized(optimized_path)
    return optimized_path


def simulate_get_dataset_item_with_insertions(input_ids_tensor, idx, dataset_insertions):
    """
    Replicate the _get_dataset_item() logic from data_loader.py.
    """
    item = {"input_ids": input_ids_tensor.clone(), "index": idx}

    if dataset_insertions is not None and idx in dataset_insertions:
        for pos, token_ids in dataset_insertions[idx]:
            end = min(pos + len(token_ids), len(item["input_ids"]))
            item["input_ids"][pos:end] = torch.tensor(
                token_ids[: end - pos], dtype=item["input_ids"].dtype
            )

    return item


def simulate_reshuffle_remap(insertion_map, global_indices):
    """
    Replicate the reshuffle() remapping logic from data_loader.py.

    Maps training-order indices -> dataset indices using global_indices.
    """
    dataset_insertions = {}
    for training_idx in insertion_map.get_all_indices():
        if training_idx < len(global_indices):
            dataset_idx = int(global_indices[training_idx])
            dataset_insertions[dataset_idx] = insertion_map.load(training_idx)
    return dataset_insertions


# ============================================================
# Test 1: Raw Token Insertion
# ============================================================
def test_raw_token_insertion():
    print("=" * 60)
    print("Test 1: Raw Token Insertion")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Parameters
        TRAINING_IDX = 5
        INSERTION_POS = 10
        INSERTION_TOKENS = [99, 98, 97]
        NUM_SEQUENCES = 100
        SEQ_LENGTH = 512

        # Create insertion map targeting training-order index 5
        optimized_path = create_optimized_insertion_map(tmpdir, {
            TRAINING_IDX: [(INSERTION_POS, INSERTION_TOKENS)]
        })

        # Load with our OLMo-core InsertionMapReader
        reader = InsertionMapReader(optimized_path)
        assert reader.has_index(TRAINING_IDX)
        assert reader.load(TRAINING_IDX) == [(INSERTION_POS, INSERTION_TOKENS)]
        print(f"  InsertionMapReader loaded correctly: {reader.get_all_indices()}")

        # Simulate global_indices (a shuffled permutation)
        rng = np.random.default_rng(42)
        global_indices = rng.permutation(NUM_SEQUENCES).astype(np.uint32)
        dataset_idx = int(global_indices[TRAINING_IDX])
        print(f"  Training-order index {TRAINING_IDX} -> dataset index {dataset_idx}")

        # Remap using our reshuffle logic
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)
        assert dataset_idx in dataset_insertions, \
            f"Expected dataset index {dataset_idx} in remapped insertions"
        assert dataset_insertions[dataset_idx] == [(INSERTION_POS, INSERTION_TOKENS)]
        print(f"  Remapped insertions: {dataset_insertions}")

        # Create a test sequence (all 1s)
        original_ids = torch.ones(SEQ_LENGTH, dtype=torch.int32)

        # Without insertions
        item_no_insert = simulate_get_dataset_item_with_insertions(
            original_ids, dataset_idx, None
        )
        assert item_no_insert["input_ids"][INSERTION_POS].item() == 1
        print(f"  Without insertions at pos {INSERTION_POS}: "
              f"{item_no_insert['input_ids'][INSERTION_POS:INSERTION_POS+3].tolist()}")

        # With insertions
        item_with_insert = simulate_get_dataset_item_with_insertions(
            original_ids, dataset_idx, dataset_insertions
        )
        actual = item_with_insert["input_ids"][INSERTION_POS:INSERTION_POS+len(INSERTION_TOKENS)].tolist()
        assert actual == INSERTION_TOKENS, f"Expected {INSERTION_TOKENS}, got {actual}"
        print(f"  With insertions at pos {INSERTION_POS}: {actual}")

        # Verify surrounding tokens are unchanged
        assert item_with_insert["input_ids"][0].item() == 1
        assert item_with_insert["input_ids"][INSERTION_POS - 1].item() == 1
        assert item_with_insert["input_ids"][INSERTION_POS + len(INSERTION_TOKENS)].item() == 1
        print(f"  Surrounding tokens unchanged: OK")

        # Verify an unaffected dataset index has no insertions
        other_idx = int(global_indices[0]) if global_indices[0] != dataset_idx else int(global_indices[1])
        item_other = simulate_get_dataset_item_with_insertions(
            original_ids, other_idx, dataset_insertions
        )
        assert item_other["input_ids"][INSERTION_POS].item() == 1
        print(f"  Unaffected sequence (dataset idx {other_idx}) unchanged: OK")

        reader.close()

    print("\n>>> Test 1 PASSED <<<\n")


# ============================================================
# Test 2: Text String Insertion (Encode/Decode Round-Trip)
# ============================================================
def test_text_encode_decode():
    print("=" * 60)
    print("Test 2: Text String Insertion (Encode/Decode Round-Trip)")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  SKIPPED: transformers not installed")
        return

    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    text = "The quick brown fox jumps over the lazy dog."
    token_ids = tokenizer.encode(text)
    print(f"  Text: {text!r}")
    print(f"  Token IDs ({len(token_ids)} tokens): {token_ids}")

    with tempfile.TemporaryDirectory() as tmpdir:
        TRAINING_IDX = 10
        TEXT_POS = 0
        NUM_SEQUENCES = 100
        SEQ_LENGTH = 512

        # Create insertion map
        optimized_path = create_optimized_insertion_map(tmpdir, {
            TRAINING_IDX: [(TEXT_POS, token_ids)]
        })

        reader = InsertionMapReader(optimized_path)

        # Simulate global indices
        rng = np.random.default_rng(123)
        global_indices = rng.permutation(NUM_SEQUENCES).astype(np.uint32)
        dataset_idx = int(global_indices[TRAINING_IDX])
        print(f"  Training-order index {TRAINING_IDX} -> dataset index {dataset_idx}")

        # Remap
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        # Apply insertions to a sequence of all 1s
        original_ids = torch.ones(SEQ_LENGTH, dtype=torch.int32)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, dataset_idx, dataset_insertions
        )

        # Extract and decode
        inserted = item["input_ids"][TEXT_POS:TEXT_POS + len(token_ids)].tolist()
        decoded = tokenizer.decode(inserted)
        print(f"  Inserted tokens: {inserted}")
        print(f"  Decoded: {decoded!r}")

        assert inserted == token_ids, f"Token mismatch: {inserted} vs {token_ids}"
        assert decoded == text, f"Text mismatch: {decoded!r} vs {text!r}"

        # Verify tokens after insertion are unchanged
        assert item["input_ids"][len(token_ids)].item() == 1
        print(f"  Tokens after insertion unchanged: OK")

        reader.close()

    print("\n>>> Test 2 PASSED <<<\n")


if __name__ == "__main__":
    test_raw_token_insertion()
    test_text_encode_decode()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)

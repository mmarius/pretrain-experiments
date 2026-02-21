"""
Tests for OLMo-core training data insertions.

Tests the cross-compatibility between pretrain-experiments' InsertionMapWriter
and OLMo-core's InsertionMapReader, plus the data loader insertion logic
(_get_dataset_item and reshuffle remapping).

Imports the OLMo-core InsertionMapReader directly from the file,
bypassing olmo_core.data.__init__.py (which pulls in DTensor and needs torch >= 2.6).

Requires the OLMo-core repo to be checked out alongside this repo.
"""

import os
import sys
import tempfile

import h5py
import numpy as np
import pytest
import torch

from pretrain_experiments.insertion_map import InsertionMapWriter
from pretrain_experiments.token_insertion import convert_insert_dict_to_index_map

# Import the OLMo-core InsertionMapReader directly from the file,
# bypassing olmo_core.data.__init__.py (which pulls in DTensor and needs torch >= 2.6).
import importlib.util

_insertion_map_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "OLMo-core", "src",
    "olmo_core", "data", "insertion_map.py"
)
if not os.path.exists(_insertion_map_path):
    pytest.skip(
        f"OLMo-core repo not found at {_insertion_map_path}",
        allow_module_level=True,
    )
_spec = importlib.util.spec_from_file_location("olmo_core_insertion_map", _insertion_map_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
InsertionMapReader = _mod.InsertionMapReader


# ============================================================
# Helpers
# ============================================================

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
            if pos + len(token_ids) > len(item["input_ids"]):
                raise RuntimeError(
                    f"Data insertion error: insertion at position {pos} with {len(token_ids)} tokens "
                    f"exceeds sequence length {len(item['input_ids'])} for dataset index {idx}. "
                    f"This indicates a bug in the insertion map construction."
                )
            end = pos + len(token_ids)
            item["input_ids"][pos:end] = torch.tensor(
                token_ids, dtype=item["input_ids"].dtype
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


def make_sequence(length, fill_value=1):
    """Create a test sequence tensor filled with a constant value."""
    return torch.full((length,), fill_value, dtype=torch.int32)


# ============================================================
# Test 1: Raw Token Insertion
# ============================================================
def test_raw_token_insertion():
    with tempfile.TemporaryDirectory() as tmpdir:
        TRAINING_IDX = 5
        INSERTION_POS = 10
        INSERTION_TOKENS = [99, 98, 97]
        NUM_SEQUENCES = 100
        SEQ_LENGTH = 512

        optimized_path = create_optimized_insertion_map(tmpdir, {
            TRAINING_IDX: [(INSERTION_POS, INSERTION_TOKENS)]
        })

        reader = InsertionMapReader(optimized_path)
        assert reader.has_index(TRAINING_IDX)
        assert reader.load(TRAINING_IDX) == [(INSERTION_POS, INSERTION_TOKENS)]

        # Simulate global_indices (a shuffled permutation)
        rng = np.random.default_rng(42)
        global_indices = rng.permutation(NUM_SEQUENCES).astype(np.uint32)
        dataset_idx = int(global_indices[TRAINING_IDX])

        # Remap using reshuffle logic
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)
        assert dataset_idx in dataset_insertions
        assert dataset_insertions[dataset_idx] == [(INSERTION_POS, INSERTION_TOKENS)]

        original_ids = make_sequence(SEQ_LENGTH)

        # Without insertions - original values preserved
        item_no_insert = simulate_get_dataset_item_with_insertions(
            original_ids, dataset_idx, None
        )
        assert item_no_insert["input_ids"][INSERTION_POS].item() == 1

        # With insertions - tokens replaced
        item_with_insert = simulate_get_dataset_item_with_insertions(
            original_ids, dataset_idx, dataset_insertions
        )
        actual = item_with_insert["input_ids"][INSERTION_POS:INSERTION_POS+len(INSERTION_TOKENS)].tolist()
        assert actual == INSERTION_TOKENS

        # Surrounding tokens unchanged
        assert item_with_insert["input_ids"][0].item() == 1
        assert item_with_insert["input_ids"][INSERTION_POS - 1].item() == 1
        assert item_with_insert["input_ids"][INSERTION_POS + len(INSERTION_TOKENS)].item() == 1

        # Unaffected sequence unchanged
        other_idx = int(global_indices[0]) if global_indices[0] != dataset_idx else int(global_indices[1])
        item_other = simulate_get_dataset_item_with_insertions(
            original_ids, other_idx, dataset_insertions
        )
        assert item_other["input_ids"][INSERTION_POS].item() == 1



# ============================================================
# Test 2: Text String Insertion (Encode/Decode Round-Trip)
# ============================================================
def test_text_encode_decode():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
    text = "The quick brown fox jumps over the lazy dog."
    token_ids = tokenizer.encode(text)

    with tempfile.TemporaryDirectory() as tmpdir:
        TRAINING_IDX = 10
        TEXT_POS = 0
        NUM_SEQUENCES = 100
        SEQ_LENGTH = 512

        optimized_path = create_optimized_insertion_map(tmpdir, {
            TRAINING_IDX: [(TEXT_POS, token_ids)]
        })

        reader = InsertionMapReader(optimized_path)

        rng = np.random.default_rng(123)
        global_indices = rng.permutation(NUM_SEQUENCES).astype(np.uint32)
        dataset_idx = int(global_indices[TRAINING_IDX])

        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        original_ids = make_sequence(SEQ_LENGTH)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, dataset_idx, dataset_insertions
        )

        inserted = item["input_ids"][TEXT_POS:TEXT_POS + len(token_ids)].tolist()
        decoded = tokenizer.decode(inserted)

        assert inserted == token_ids
        assert decoded == text
        assert item["input_ids"][len(token_ids)].item() == 1



# ============================================================
# Test 3: Multiple Insertions in One Sequence
# ============================================================
def test_multiple_insertions_per_sequence():
    """Verify that multiple insertions in the same sequence are all applied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 512
        TRAINING_IDX = 0
        insertions = [
            (10, [50, 51, 52]),
            (100, [60, 61]),
            (200, [70, 71, 72, 73]),
        ]

        optimized_path = create_optimized_insertion_map(tmpdir, {
            TRAINING_IDX: insertions
        })

        reader = InsertionMapReader(optimized_path)
        loaded = reader.load(TRAINING_IDX)
        assert loaded == insertions

        # Identity global_indices (no shuffle)
        global_indices = np.arange(100, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        original_ids = make_sequence(SEQ_LENGTH)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, TRAINING_IDX, dataset_insertions
        )

        # Each insertion should be present
        for pos, tokens in insertions:
            actual = item["input_ids"][pos:pos+len(tokens)].tolist()
            assert actual == tokens, f"At pos {pos}: expected {tokens}, got {actual}"

        # Positions between insertions should be unchanged
        assert item["input_ids"][0].item() == 1
        assert item["input_ids"][50].item() == 1
        assert item["input_ids"][150].item() == 1
        assert item["input_ids"][300].item() == 1



# ============================================================
# Test 4: Insertion Exceeding Sequence Length Raises Error
# ============================================================
def test_insertion_exceeding_sequence_raises_error():
    """Insertion that extends past the end of the sequence raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 32
        TRAINING_IDX = 0
        # Insert 10 tokens starting at position 28 -> would exceed sequence length
        insertion_tokens = list(range(90, 100))  # 10 tokens
        insertion_pos = 28

        optimized_path = create_optimized_insertion_map(tmpdir, {
            TRAINING_IDX: [(insertion_pos, insertion_tokens)]
        })

        reader = InsertionMapReader(optimized_path)
        global_indices = np.arange(10, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        original_ids = make_sequence(SEQ_LENGTH)
        with pytest.raises(RuntimeError, match="exceeds sequence length"):
            simulate_get_dataset_item_with_insertions(
                original_ids, TRAINING_IDX, dataset_insertions
            )



# ============================================================
# Test 5: Full Pipeline (insert_dict -> convert -> write -> read -> apply)
# ============================================================
def test_full_pipeline_from_insert_dict():
    """
    End-to-end test: global insert_dict -> convert_insert_dict_to_index_map
    -> InsertionMapWriter -> OLMo-core InsertionMapReader -> data loader simulation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 4096
        NUM_SEQUENCES = 50

        # Build a global insert_dict (as pretrain_experiment.py would)
        # Insert at various global positions
        insert_dict = {
            100: [10, 20, 30],                      # seq 0, local pos 100
            4096 + 50: [40, 50],                     # seq 1, local pos 50
            2 * 4096 + 4000: [60, 70, 80, 90],       # seq 2, local pos 4000
        }

        # Convert to index map (same as OLMoCoreFramework.set_experiments)
        index_map = convert_insert_dict_to_index_map(
            insert_dict,
            num_index_tokens=SEQ_LENGTH,
            split_across_boundaries=False,
        )

        # Verify the conversion
        assert 0 in index_map
        assert 1 in index_map
        assert 2 in index_map
        assert index_map[0] == [(100, [10, 20, 30])]
        assert index_map[1] == [(50, [40, 50])]
        assert index_map[2] == [(4000, [60, 70, 80, 90])]

        # Write and optimize
        optimized_path = create_optimized_insertion_map(tmpdir, index_map)

        # Read with OLMo-core reader
        reader = InsertionMapReader(optimized_path)
        assert len(reader.get_all_indices()) == 3

        # Simulate with identity mapping (no shuffle)
        global_indices = np.arange(NUM_SEQUENCES, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        # Apply to each sequence and verify
        for seq_idx in range(3):
            original_ids = make_sequence(SEQ_LENGTH)
            item = simulate_get_dataset_item_with_insertions(
                original_ids, seq_idx, dataset_insertions
            )

            # Verify the expected insertions for this sequence
            for pos, tokens in index_map[seq_idx]:
                actual = item["input_ids"][pos:pos+len(tokens)].tolist()
                assert actual == tokens, \
                    f"Seq {seq_idx}, pos {pos}: expected {tokens}, got {actual}"

        # Verify sequences without insertions are untouched
        original_ids = make_sequence(SEQ_LENGTH)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, 5, dataset_insertions
        )
        assert (item["input_ids"] == 1).all()



# ============================================================
# Test 6: Insertion at Position 0
# ============================================================
def test_insertion_at_position_zero():
    """Verify insertion at the very start of a sequence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 64
        TOKENS = [42, 43, 44, 45]

        optimized_path = create_optimized_insertion_map(tmpdir, {
            0: [(0, TOKENS)]
        })

        reader = InsertionMapReader(optimized_path)
        global_indices = np.arange(10, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        original_ids = make_sequence(SEQ_LENGTH)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, 0, dataset_insertions
        )

        assert item["input_ids"][:len(TOKENS)].tolist() == TOKENS
        assert item["input_ids"][len(TOKENS)].item() == 1



# ============================================================
# Test 7: Insertion at Last Valid Position
# ============================================================
def test_insertion_at_last_position():
    """Verify single-token insertion at the very last position."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 64
        LAST_POS = SEQ_LENGTH - 1
        TOKEN = [99]

        optimized_path = create_optimized_insertion_map(tmpdir, {
            0: [(LAST_POS, TOKEN)]
        })

        reader = InsertionMapReader(optimized_path)
        global_indices = np.arange(10, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        original_ids = make_sequence(SEQ_LENGTH)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, 0, dataset_insertions
        )

        assert item["input_ids"][LAST_POS].item() == 99
        assert item["input_ids"][LAST_POS - 1].item() == 1



# ============================================================
# Test 8: Single-Token Insertion
# ============================================================
def test_single_token_insertion():
    """Verify a single token can be inserted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 128
        POS = 64

        optimized_path = create_optimized_insertion_map(tmpdir, {
            3: [(POS, [777])]
        })

        reader = InsertionMapReader(optimized_path)
        global_indices = np.arange(10, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        original_ids = make_sequence(SEQ_LENGTH)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, 3, dataset_insertions
        )

        assert item["input_ids"][POS].item() == 777
        assert item["input_ids"][POS - 1].item() == 1
        assert item["input_ids"][POS + 1].item() == 1



# ============================================================
# Test 9: Large Insertion (many tokens)
# ============================================================
def test_large_token_insertion():
    """Verify a large insertion (e.g. 1000 tokens) works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 4096
        POS = 100
        TOKENS = list(range(2000, 3000))  # 1000 tokens

        optimized_path = create_optimized_insertion_map(tmpdir, {
            0: [(POS, TOKENS)]
        })

        reader = InsertionMapReader(optimized_path)
        global_indices = np.arange(10, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        original_ids = make_sequence(SEQ_LENGTH)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, 0, dataset_insertions
        )

        actual = item["input_ids"][POS:POS+len(TOKENS)].tolist()
        assert actual == TOKENS
        assert item["input_ids"][POS - 1].item() == 1
        assert item["input_ids"][POS + len(TOKENS)].item() == 1



# ============================================================
# Test 10: Many Sequences with Insertions
# ============================================================
def test_many_sequences_with_insertions():
    """Verify insertions across many different sequences."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 256
        NUM_SEQUENCES = 1000
        NUM_INSERTIONS = 200

        # Create insertions for 200 different sequences
        insertion_dict = {}
        rng = np.random.default_rng(99)
        for i in range(NUM_INSERTIONS):
            seq_idx = i * 5  # spread across sequences
            pos = int(rng.integers(0, SEQ_LENGTH - 10))
            tokens = list(range(i * 10, i * 10 + 5))  # 5 tokens each
            insertion_dict[seq_idx] = [(pos, tokens)]

        optimized_path = create_optimized_insertion_map(tmpdir, insertion_dict)

        reader = InsertionMapReader(optimized_path)
        assert len(reader.get_all_indices()) == NUM_INSERTIONS

        # Identity mapping
        global_indices = np.arange(NUM_SEQUENCES, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        # Spot-check a few
        for seq_idx in [0, 50, 100, 500, 995]:
            original_ids = make_sequence(SEQ_LENGTH)
            item = simulate_get_dataset_item_with_insertions(
                original_ids, seq_idx, dataset_insertions
            )
            if seq_idx in dataset_insertions:
                for pos, tokens in dataset_insertions[seq_idx]:
                    actual = item["input_ids"][pos:pos+len(tokens)].tolist()
                    assert actual == tokens



# ============================================================
# Test 11: No-Op Cases
# ============================================================
def test_no_insertions_for_index():
    """Sequence with no insertions should be completely unchanged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 128

        # Only insert into index 5
        optimized_path = create_optimized_insertion_map(tmpdir, {
            5: [(10, [99, 98])]
        })

        reader = InsertionMapReader(optimized_path)
        assert not reader.has_index(0)
        assert reader.load(0) is None

        global_indices = np.arange(10, dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        # Index 0 should have no insertions
        original_ids = make_sequence(SEQ_LENGTH, fill_value=42)
        item = simulate_get_dataset_item_with_insertions(
            original_ids, 0, dataset_insertions
        )
        assert (item["input_ids"] == 42).all(), "Sequence without insertions should be unchanged"



def test_empty_dataset_insertions():
    """When dataset_insertions is None, no changes should be made."""
    SEQ_LENGTH = 128
    original_ids = make_sequence(SEQ_LENGTH, fill_value=7)
    item = simulate_get_dataset_item_with_insertions(original_ids, 0, None)
    assert (item["input_ids"] == 7).all()


# ============================================================
# Test 12: Reshuffle Correctness
# ============================================================
def test_reshuffle_maps_correctly():
    """
    Verify that the reshuffle logic correctly maps training-order indices
    to dataset indices using global_indices permutation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Insert into training-order indices 0, 2, 4
        insertion_dict = {
            0: [(10, [100])],
            2: [(20, [200])],
            4: [(30, [300])],
        }

        optimized_path = create_optimized_insertion_map(tmpdir, insertion_dict)
        reader = InsertionMapReader(optimized_path)

        # Create a known permutation: [3, 1, 4, 0, 2]
        # training_idx 0 -> dataset_idx 3
        # training_idx 2 -> dataset_idx 4
        # training_idx 4 -> dataset_idx 2
        global_indices = np.array([3, 1, 4, 0, 2], dtype=np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        assert 3 in dataset_insertions  # training 0 -> dataset 3
        assert 4 in dataset_insertions  # training 2 -> dataset 4
        assert 2 in dataset_insertions  # training 4 -> dataset 2

        assert dataset_insertions[3] == [(10, [100])]
        assert dataset_insertions[4] == [(20, [200])]
        assert dataset_insertions[2] == [(30, [300])]

        # Original training indices should NOT be in dataset_insertions
        # (unless they happen to be mapped there, which 2 is)
        assert 0 not in dataset_insertions
        # 1 not in dataset_insertions
        assert 1 not in dataset_insertions



# ============================================================
# Test 13: Cross-Compatibility Writer/Reader
# ============================================================
def test_writer_reader_cross_compatibility_edge_cases():
    """
    Test that pretrain-experiments Writer and OLMo-core Reader agree
    on edge cases: empty token lists, very large token IDs, etc.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        insertion_dict = {
            0: [(0, [0])],                          # minimal: token ID 0 at pos 0
            1: [(100, [2**31 - 1])],                 # large token ID (max int32)
            2: [(50, list(range(500)))],              # 500 tokens
            100: [(0, [1, 2]), (10, [3, 4, 5])],      # two insertions
        }

        optimized_path = create_optimized_insertion_map(tmpdir, insertion_dict)
        reader = InsertionMapReader(optimized_path)

        # Verify each entry round-trips correctly
        for idx, expected in insertion_dict.items():
            loaded = reader.load(idx)
            assert loaded == expected, \
                f"Index {idx}: expected {expected}, got {loaded}"

        # Non-existent index
        assert reader.load(999) is None
        assert not reader.has_index(999)



# ============================================================
# Test 14: Original Tensor Not Modified
# ============================================================
def test_original_tensor_not_modified():
    """Verify that the simulation clones the tensor (original is not mutated)."""
    SEQ_LENGTH = 64
    original_ids = make_sequence(SEQ_LENGTH, fill_value=1)
    original_clone = original_ids.clone()

    dataset_insertions = {0: [(10, [99, 98, 97])]}
    item = simulate_get_dataset_item_with_insertions(
        original_ids, 0, dataset_insertions
    )

    # The returned item should have the insertion
    assert item["input_ids"][10].item() == 99
    # But the original tensor should be untouched
    assert (original_ids == original_clone).all(), "Original tensor was modified!"


# ============================================================
# Test 15: Full Pipeline with Shuffled Global Indices
# ============================================================
def test_full_pipeline_with_shuffle():
    """
    End-to-end test with actual shuffling, verifying that each insertion
    ends up in the correct dataset sequence after remapping.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        SEQ_LENGTH = 512
        NUM_SEQUENCES = 200

        # Global insert_dict with positions in different sequences
        insert_dict = {
            0: [10, 20, 30],                         # seq 0, pos 0
            512 * 5 + 100: [40, 50, 60],              # seq 5, pos 100
            512 * 99 + 400: [70, 80],                  # seq 99, pos 400
        }

        index_map = convert_insert_dict_to_index_map(
            insert_dict, num_index_tokens=SEQ_LENGTH, split_across_boundaries=False
        )

        optimized_path = create_optimized_insertion_map(tmpdir, index_map)
        reader = InsertionMapReader(optimized_path)

        # Shuffle
        rng = np.random.default_rng(2024)
        global_indices = rng.permutation(NUM_SEQUENCES).astype(np.uint32)
        dataset_insertions = simulate_reshuffle_remap(reader, global_indices)

        # For each training index with insertions, verify the corresponding
        # dataset index gets the correct tokens
        for training_idx, expected_insertions in index_map.items():
            dataset_idx = int(global_indices[training_idx])
            assert dataset_idx in dataset_insertions

            original_ids = make_sequence(SEQ_LENGTH)
            item = simulate_get_dataset_item_with_insertions(
                original_ids, dataset_idx, dataset_insertions
            )

            for pos, tokens in expected_insertions:
                actual = item["input_ids"][pos:pos+len(tokens)].tolist()
                assert actual == tokens, \
                    f"Training idx {training_idx} -> dataset idx {dataset_idx}, " \
                    f"pos {pos}: expected {tokens}, got {actual}"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])

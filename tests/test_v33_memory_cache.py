import torch

from flued.v33.memory import CausalLowRankSequenceMemory, MemoryState


def test_committed_kv_cache_matches_uncached_read() -> None:
    torch.manual_seed(7)
    memory = CausalLowRankSequenceMemory(query_dim=8, d_mem=8, top_k=3)
    committed = torch.randn(2, 5, 8)
    committed_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    z_query = torch.randn(2, 4, 8)
    uncached = MemoryState(committed=committed, committed_mask=committed_mask)

    with torch.no_grad():
        cached = MemoryState(
            committed=committed,
            committed_mask=committed_mask,
            committed_key=memory.key(committed),
            committed_value=memory.value(committed),
        )
        uncached_read = memory.read(z_query, uncached)
        cached_read = memory.read(z_query, cached)

    assert torch.allclose(cached_read, uncached_read, atol=1.0e-6)


def test_committed_kv_cache_matches_uncached_causal_read_with_current_write() -> None:
    torch.manual_seed(11)
    memory = CausalLowRankSequenceMemory(query_dim=8, d_mem=8, top_k=3)
    committed = torch.randn(1, 6, 8)
    committed_mask = torch.ones(1, 6, dtype=torch.bool)
    z_query = torch.randn(1, 4, 8)
    current_write = torch.randn(1, 4, 2, 8)
    chunk_mask = torch.tensor([[True, True, True, False]])
    uncached = MemoryState(committed=committed, committed_mask=committed_mask)

    with torch.no_grad():
        cached = MemoryState(
            committed=committed,
            committed_mask=committed_mask,
            committed_key=memory.key(committed),
            committed_value=memory.value(committed),
        )
        uncached_read, uncached_aux = memory.read_causal(z_query, current_write, chunk_mask, uncached)
        cached_read, cached_aux = memory.read_causal(z_query, current_write, chunk_mask, cached)

    assert torch.allclose(cached_read, uncached_read, atol=1.0e-6)
    assert torch.equal(cached_aux["has_memory"], uncached_aux["has_memory"])


def test_bidirectional_no_self_reads_other_current_chunks_only() -> None:
    torch.manual_seed(12)
    memory = CausalLowRankSequenceMemory(query_dim=8, d_mem=8, top_k=6)
    z_query = torch.randn(1, 4, 8)
    current_write = torch.randn(1, 4, 2, 8)
    chunk_mask = torch.tensor([[True, True, True, False]])

    _read, aux = memory.read_with_visibility(
        z_query,
        current_write,
        chunk_mask,
        state=None,
        visibility="bidirectional_no_self",
    )

    assert torch.equal(aux["has_memory"], torch.tensor([[True, True, True, False]]))
    assert torch.equal(aux["self_allowed_count"], torch.zeros_like(aux["self_allowed_count"]))
    # Three active chunks, rank two each, excluding self leaves four visible
    # local memory slots per active query chunk.
    assert torch.equal(aux["visible_slots"][0, :3], torch.full((3,), 4))


def test_commit_caches_only_in_no_grad_path() -> None:
    torch.manual_seed(13)
    memory = CausalLowRankSequenceMemory(query_dim=8, d_mem=8, top_k=2)
    m_write = torch.randn(1, 3, 2, 8)
    chunk_mask = torch.tensor([[True, True, False]])

    train_state = memory.write(m_write, chunk_mask, None)
    train_committed = memory.commit(train_state)
    assert train_committed.committed_key is None
    assert train_committed.committed_value is None

    with torch.no_grad():
        infer_state = memory.write(m_write, chunk_mask, None)
        infer_committed = memory.commit(infer_state)

    assert infer_committed.committed_key is not None
    assert infer_committed.committed_value is not None
    assert infer_committed.committed_key.shape == infer_committed.committed.shape
    assert infer_committed.committed_value.shape == infer_committed.committed.shape

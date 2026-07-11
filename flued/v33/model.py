from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flued.data import PAD_ID

from .byte_lookup import StructuredByteLookup
from .chunk_builder import ChunkBatch, ChunkBuilder
from .decoder import SharedParameterDecoder
from .interpreter import InterpreterOutput, LatentMemoryInterpreter
from .memory import CausalLowRankSequenceMemory, MemoryState
from .segmentor import SegmentorOutput, SignedBoundarySegmentor
from .threshold_policy import DualThresholdPolicy, ThresholdPolicyOutput


@dataclass
class FLUEDV33Config:
    d_model: int = 256
    d_z: int = 256
    d_mem: int = 256
    hidden: int = 512
    max_chunks: int = 128
    max_span: int = 16
    use_memory: bool = False
    memory_rank: int = 0
    memory_top_k: int = 4
    memory_build_mode: str = "parallel_local"
    memory_visibility: str = "bidirectional_no_self"
    max_readout_vectors: int = 1
    chunk_mixer: str = "mean"
    tau_cut: float = 0.90
    tau_trans: float = 0.75
    tau_keep: float = 0.65


@dataclass
class FLUEDV33Output:
    byte_logits: torch.Tensor
    length_logits: torch.Tensor
    z_content: torch.Tensor
    readout_z: torch.Tensor
    readout_gate: torch.Tensor
    readout_emit: torch.Tensor
    chunks: ChunkBatch
    segmentor: SegmentorOutput
    policy: ThresholdPolicyOutput
    interpreter: InterpreterOutput
    memory_state: MemoryState | None
    aux: dict


class FLUEDV33(nn.Module):
    """FLUED v3.3 byte-to-latent decision-interface skeleton."""

    def __init__(self, config: FLUEDV33Config | None = None) -> None:
        super().__init__()
        self.config = config or FLUEDV33Config()
        if self.config.memory_build_mode not in {"causal_current", "parallel_local"}:
            raise ValueError(f"unknown memory_build_mode: {self.config.memory_build_mode}")
        if self.config.memory_visibility not in {"past_only", "bidirectional_no_self", "all_visible"}:
            raise ValueError(f"unknown memory_visibility: {self.config.memory_visibility}")
        self.byte_lookup = StructuredByteLookup(self.config.d_model)
        self.segmentor = SignedBoundarySegmentor(self.config.d_model, hidden=self.config.hidden)
        self.policy = DualThresholdPolicy(self.config.tau_cut, self.config.tau_trans, self.config.tau_keep)
        self.chunk_builder = ChunkBuilder(self.config.max_chunks, self.config.max_span)
        memory_rank = self.config.memory_rank if self.config.use_memory else 0
        self.interpreter = LatentMemoryInterpreter(
            self.config.d_model,
            self.config.d_z,
            d_mem=self.config.d_mem,
            memory_rank=memory_rank,
            max_readout_vectors=self.config.max_readout_vectors,
            chunk_mixer=self.config.chunk_mixer,
        )
        self.memory = (
            CausalLowRankSequenceMemory(self.config.d_model, self.config.d_mem, top_k=self.config.memory_top_k)
            if self.config.use_memory
            else None
        )
        self.decoder = SharedParameterDecoder(
            self.config.d_z,
            self.config.hidden,
            self.config.max_span,
            byte_lookup=self.byte_lookup,
            d_model=self.config.d_model,
        )

    def forward(self, token_ids: torch.Tensor, memory_state: MemoryState | None = None, commit_memory: bool = True) -> FLUEDV33Output:
        valid = token_ids.ne(PAD_ID)
        byte_features = self.byte_lookup(token_ids)
        segmentor = self.segmentor(byte_features, valid)
        policy = self.policy(segmentor.confidence, valid)
        chunks = self.chunk_builder(
            byte_features,
            valid,
            policy.hard_cut,
            confidence=segmentor.confidence,
            soft_transition=policy.soft_transition,
            force_continue=policy.force_continue,
        )

        memory_read = None
        memory_aux = {}
        current_write = None
        pooled = None
        if self.memory is not None:
            pooled = self.interpreter.mean_pool(chunks)
            current_write = self.interpreter.write_memory(chunks, pooled=pooled)
            visibility = "past_only" if self.config.memory_build_mode == "causal_current" else self.config.memory_visibility
            memory_read, memory_aux = self.memory.read_with_visibility(
                pooled,
                current_write,
                chunks.chunk_mask,
                memory_state,
                visibility=visibility,
            )

        interpreter = self.interpreter(chunks, memory_read=memory_read, pooled=pooled, m_write=current_write)
        next_memory_state = memory_state
        if self.memory is not None:
            next_memory_state = self.memory.write(interpreter.m_write, chunks.chunk_mask, memory_state)
            if commit_memory:
                next_memory_state = self.memory.commit(next_memory_state)

        byte_logits, length_logits = self.decoder(interpreter.readout_z, chunks.chunk_mask, readout_gate=interpreter.readout_gate)
        bytes_n = valid.float().sum().clamp(min=1.0)
        readout_units = interpreter.aux["readout_units"]
        emit_units = interpreter.aux["emit_units"]
        memory_units = readout_units.new_zeros(())
        if next_memory_state is not None and next_memory_state.committed_mask is not None:
            memory_units = next_memory_state.committed_mask.float().sum()
        aux = {
            "readout_units_per_byte": readout_units / bytes_n,
            "emit_units_per_byte": emit_units / bytes_n,
            "memory_units": memory_units,
            "memory_enabled": self.memory is not None,
            "truncated_tokens": chunks.pack_info.get("truncated_tokens"),
            "memory_has_context": memory_aux.get("has_memory"),
            "memory_read_entropy": memory_aux.get("read_entropy"),
            "memory_read_norm": memory_aux.get("read_norm"),
            "memory_self_allowed_count": memory_aux.get("self_allowed_count"),
            "memory_visible_slots": memory_aux.get("visible_slots"),
            "memory_gate": interpreter.aux.get("memory_gate"),
            "readout_gate_mean": interpreter.aux.get("readout_gate_mean"),
            "readout_emit_mean": interpreter.aux.get("readout_emit_mean"),
            "active_chunks_per_byte": chunks.chunk_mask.float().sum() / bytes_n,
        }
        return FLUEDV33Output(
            byte_logits=byte_logits,
            length_logits=length_logits,
            z_content=interpreter.z_content,
            readout_z=interpreter.readout_z,
            readout_gate=interpreter.readout_gate,
            readout_emit=interpreter.readout_emit,
            chunks=chunks,
            segmentor=segmentor,
            policy=policy,
            interpreter=interpreter,
            memory_state=next_memory_state,
            aux=aux,
        )

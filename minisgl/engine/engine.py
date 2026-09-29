"""Core inference engine: holds model, KV cache, and runs forward + sampling."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.config import ModelArgs, ServerArgs
    from minisgl.models.attention.metadata import AttentionMetadata
    from minisgl.scheduler.batch import Batch

__all__ = ["Engine"]
import torch

from minisgl.engine import model_runner
from minisgl.models.attention.dispatcher import AttentionBackend
from minisgl.models.attention.layer import BaseAttention
from minisgl.sampling import Sampler
from minisgl.utils.device import (
    get_device,
    get_device_type,
    get_tp_rank,
    init_distributed,
    synchronize,
)
from minisgl.utils.logger import logger


class Engine:
    """Core inference engine.

    Holds the model, KV cache pool, and runs forward + sampling. Decode
    optionally runs on captured execution graphs (see GraphRunner).
    Supports CUDA, NPU (Ascend), and CPU devices.
    Supports context manager protocol for resource cleanup.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        model_args: ModelArgs,
        tp_rank: int = 0,
    ) -> None:
        self.server_args = server_args
        self.model_args = model_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        if self.tp_size > 1 and server_args.cuda_graph_bs:
            raise ValueError("CUDA graphs are not supported with tp_size > 1 yet")
        self._device_type = get_device_type()

        self._maybe_init_distributed(model_args)
        from minisgl.engine.collectives import configure

        configure(server_args.tp_collective_mode, server_args.collective_stats)
        # Distributed initialization selects the local CUDA/NPU device. Do not
        # cache a device before that point: spawned workers would otherwise all
        # construct their model on device 0.
        self.device = get_device()
        AttentionBackend.configure(server_args.attention_backend)
        self._clamp_max_seq_len(model_args)
        self._validate_model_path(server_args.model_path)

        # --- Model assembly (delegated to model_runner for testability) ---
        model, model_type = model_runner.detect_and_create_model(
            server_args.model_path, model_args
        )
        self._model_type = model_type
        self.dtype = model_runner.resolve_dtype(
            server_args.dtype, server_args.model_path, self.device.type
        )
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        # No autograd during capture/replay (or any inference forward).
        model.requires_grad_(False)

        model_runner.prebuild_rope(model, server_args.max_seq_len, self.device)

        loaded = model_runner.load_model_weights(
            model, server_args.model_path, self.tp_rank, self.tp_size
        )
        logger.info("Loaded %d weights (model_type=%s)", loaded, self._model_type)

        # --- KV cache allocation (delegated to KVCacheAllocator) ---
        from minisgl.engine.kvcache.allocator import KVCacheAllocator

        allocator = KVCacheAllocator(server_args, model_args)
        self.kv_cache_pool = allocator.allocate(model, self.device, self.tp_size)

        # --- Execution + sampling (delegated to ModelRunner / Sampler) ---
        self.runner = model_runner.ModelRunner(model, server_args, self.device)
        self.model = model  # public alias kept for tests / GraphRunner coupling
        self.batch_context = self.runner.batch_context
        # GraphRunner is Engine-owned: it needs the KV cache pool and Engine's
        # execution/device aliases, all of which are ready only at this point.
        self.graph_runner = None
        if (
            server_args.cuda_graph_bs
            and server_args.cuda_graph_bs > 0
            and self.device.type in ("cuda", "npu")
        ):
            from minisgl.engine.graph import GraphRunner

            self.graph_runner = GraphRunner(self)
            self.runner.graph_runner = self.graph_runner

        self.sampler = Sampler()

        logger.info(
            "Engine initialized on rank %d (device=%s)", tp_rank, self._device_type
        )

    def _maybe_init_distributed(self, model_args: ModelArgs) -> None:
        """Initialize the process group when tensor parallelism is requested."""
        if self.tp_size <= 1:
            return
        if not 2 <= self.tp_size <= 8:
            raise ValueError("tp_size must be between 1 and 8")
        init_distributed(tp_rank=self.tp_rank, tp_size=self.tp_size)
        dimensions = {
            "num_attention_heads": model_args.num_attention_heads,
            "num_kv_heads": model_args.num_kv_heads,
            "intermediate_size": model_args.intermediate_size,
            "vocab_size": model_args.vocab_size,
        }
        if model_args.num_experts:
            raise ValueError("Qwen3-MoE tensor parallelism is not supported yet")
        invalid = [name for name, value in dimensions.items() if value % self.tp_size]
        if invalid:
            raise ValueError(
                f"Model dimensions must be divisible by tp_size={self.tp_size}: "
                + ", ".join(invalid)
            )

    def _clamp_max_seq_len(self, model_args: ModelArgs) -> None:
        """Clamp max_seq_len to the model's trained context window.

        RoPE tables and the paged KV pool / req_to_token buffers are all sized
        off this value, so an oversized --max-seq-len would only surface later
        as an index error mid-generation. Engine is always built before the
        Scheduler (which shares server_args), so clamping here propagates
        everywhere.
        """
        args = self.server_args
        max_pos = model_args.max_position_embeddings
        if max_pos and args.max_seq_len > max_pos:
            logger.warning(
                "max_seq_len=%d exceeds the model's max_position_embeddings=%d; "
                "clamping to %d",
                args.max_seq_len,
                max_pos,
                max_pos,
            )
            args.max_seq_len = max_pos

    @staticmethod
    def _validate_model_path(model_path: str) -> None:
        """Fail fast with a clear message when the model dir is unusable.

        Catches the common typo / un-downloaded case up front instead of an
        opaque FileNotFoundError deep inside registry/config loading.
        """
        from pathlib import Path

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {model_path!r}. "
                "Pass a directory containing config.json."
            )
        if not p.is_dir():
            raise NotADirectoryError(
                f"Model path is not a directory: {model_path!r}."
            )
        if not (p / "config.json").is_file():
            raise FileNotFoundError(
                f"Model path has no config.json: {model_path!r}. "
                "This does not look like a HuggingFace model directory."
            )

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Release accelerator resources."""
        if self.graph_runner is not None:
            self.graph_runner.clear()
        # The per-layer k_cache/v_cache references are views into the pool
        # buffer; drop them too or the pool's memory stays referenced and
        # cannot be released. The graph pad handle is released together with
        # the pool.
        if hasattr(self, "model"):
            for module in self.model.modules():
                if isinstance(module, BaseAttention):
                    module.k_cache = None
                    module.v_cache = None
        if hasattr(self, "kv_cache_pool"):
            del self.kv_cache_pool
        synchronize()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()

    def _run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_meta: AttentionMetadata | None = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single forward entry (alias to ModelRunner, kept for GraphRunner)."""
        return self.runner._run_model(
            input_ids, positions, attn_meta, logits_indices
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Run a model forward pass on a scheduler batch (via ModelRunner)."""
        self._set_logits_gather(self._can_distribute_sampling(batch))
        return self.runner.forward(batch)

    def _can_distribute_sampling(self, batch: Batch) -> bool:
        """Return whether this batch can use bounded local-vocab sampling."""
        if self.tp_size <= 1 or self.server_args.tp_sampling_mode not in (
            "distributed",
            "distributed_packed",
        ):
            return False
        return all(
            req.sampling_params.temperature <= 0.0 or req.sampling_params.top_k > 0
            for req in batch.reqs
        )

    def _set_logits_gather(self, distributed: bool) -> None:
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:
            # The default remains True, preserving the original full-logits path.
            lm_head.gather_output = not distributed

    def sample(self, logits: torch.Tensor, batch: Batch) -> list[int]:
        """Sample next tokens from logits.

        Groups requests with identical sampling params for batched sampling
        (see Sampler.sample_batch).
        """
        # Normalize to (num_reqs, vocab_size): decode may produce
        # (num_reqs, 1, vocab_size) or, for a single request, (vocab_size,).
        # Prefill already yields (num_reqs, vocab_size) — forward gathered each
        # request's last-position logits via logits_indices, so the
        # normalization below is a no-op for it.
        if logits.dim() == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)  # (num_reqs, vocab_size)
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0)  # (1, vocab_size)

        params = [req.sampling_params for req in batch.reqs]
        if self.tp_size <= 1:
            return self.sampler.sample_batch(logits, params)

        # Sampling is a control-plane operation. Only rank 0 consumes random
        # state; every rank receives exactly the same token IDs before the
        # scheduler mutates request/KV state.
        distributed = self._can_distribute_sampling(batch)
        if distributed:
            if self.server_args.tp_sampling_mode == "distributed_packed":
                tokens = self._sample_distributed_packed(logits, params)
            else:
                tokens = self._sample_distributed(logits, params)
        else:
            tokens = self.sampler.sample_batch(logits, params) if get_tp_rank() == 0 else None
        token_tensor = torch.empty(len(batch.reqs), dtype=torch.long, device=self.device)
        if tokens is not None:
            token_tensor.copy_(torch.tensor(tokens, dtype=torch.long, device=self.device))
        from minisgl.engine.collectives import broadcast

        broadcast(token_tensor, src=0)
        return token_tensor.tolist()

    def _sample_distributed(self, logits: torch.Tensor, params: list) -> list[int] | None:
        """Sample from globally bounded candidates without full-vocab gather."""
        from minisgl.engine.collectives import all_gather

        vocab_per_rank = self.model_args.vocab_size // self.tp_size
        max_k = max((max(1, p.top_k) if p.temperature > 0 else 1) for p in params)
        max_k = min(max_k, logits.shape[-1])
        values, indices = logits.topk(max_k, dim=-1)
        indices = indices + get_tp_rank() * vocab_per_rank

        # Candidate rows are gathered along the last dimension, so the result
        # is [batch, tp_size * k] and can be passed to the existing sampler.
        values = all_gather(values, dim=-1)
        indices = all_gather(indices, dim=-1)
        if get_tp_rank() != 0:
            return None
        sampled = self.sampler.sample_batch(values, params)
        return (
            indices.gather(1, torch.tensor(sampled, device=self.device).view(-1, 1))
            .squeeze(1)
            .tolist()
        )

    @staticmethod
    def _pack_sampling_candidates(
        values: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Pack heterogeneous candidate tensors without changing their bits."""
        batch = values.shape[0]
        value_bytes = values.element_size() * values.shape[1]
        index_bytes = indices.element_size() * indices.shape[1]
        value_packet = values.contiguous().view(torch.uint8).view(batch, value_bytes)
        index_packet = indices.contiguous().view(torch.uint8).view(batch, index_bytes)
        return torch.cat((value_packet, index_packet), dim=-1), value_bytes, index_bytes

    @staticmethod
    def _unpack_sampling_candidates(
        packet: torch.Tensor,
        value_dtype: torch.dtype,
        index_dtype: torch.dtype,
        k: int,
        value_bytes: int,
        index_bytes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore gathered candidates from a byte packet."""
        packet_width = value_bytes + index_bytes
        chunks = packet.split(packet_width, dim=-1)
        values = []
        indices = []
        for chunk in chunks:
            values.append(chunk[..., :value_bytes].contiguous().view(value_dtype).view(-1, k))
            indices.append(chunk[..., value_bytes:].contiguous().view(index_dtype).view(-1, k))
        return torch.cat(values, dim=-1), torch.cat(indices, dim=-1)

    def _sample_distributed_packed(
        self, logits: torch.Tensor, params: list
    ) -> list[int] | None:
        """Sample bounded candidates with one byte-packed all-gather."""
        from minisgl.engine.collectives import all_gather

        vocab_per_rank = self.model_args.vocab_size // self.tp_size
        max_k = max((max(1, p.top_k) if p.temperature > 0 else 1) for p in params)
        max_k = min(max_k, logits.shape[-1])
        values, indices = logits.topk(max_k, dim=-1)
        indices = indices + get_tp_rank() * vocab_per_rank
        packet, value_bytes, index_bytes = self._pack_sampling_candidates(values, indices)
        packet = all_gather(packet, dim=-1)
        values, indices = self._unpack_sampling_candidates(
            packet,
            values.dtype,
            indices.dtype,
            max_k,
            value_bytes,
            index_bytes,
        )
        if get_tp_rank() != 0:
            return None
        sampled = self.sampler.sample_batch(values, params)
        return (
            indices.gather(1, torch.tensor(sampled, device=self.device).view(-1, 1))
            .squeeze(1)
            .tolist()
        )

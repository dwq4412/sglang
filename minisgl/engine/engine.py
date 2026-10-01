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

        configure(
            server_args.tp_collective_mode,
            server_args.collective_stats,
            server_args.tp_ar_chunk_bytes,
        )
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
            raise NotADirectoryError(f"Model path is not a directory: {model_path!r}.")
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
        graph_runner = getattr(self, "graph_runner", None)
        if graph_runner is not None:
            graph_runner.clear()
            # GraphRunner keeps references to Engine and _run_model. Break
            # that cycle before dropping the model and runner below.
            graph_runner.engine = None
            graph_runner._run_model = None
        self.graph_runner = None
        # The per-layer k_cache/v_cache references are views into the pool
        # buffer; drop them too or the pool's memory stays referenced and
        # cannot be released. The graph pad handle is released together with
        # the pool.
        model = getattr(self, "model", None)
        if model is not None:
            for module in model.modules():
                if isinstance(module, BaseAttention):
                    module.k_cache = None
                    module.v_cache = None
            del model
        self.kv_cache_pool = None

        # ModelRunner owns another strong reference to the model and may own
        # device-resident batch metadata. Detach every Engine-owned reference
        # so the next benchmark iteration cannot inherit the previous model's
        # weights through the runner.
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.graph_runner = None
            runner.model = None
            runner.batch_context = None
            del runner
        self.runner = None
        self.batch_context = None
        self.model = None
        synchronize()
        if getattr(self, "_device_type", None) == "cuda":
            torch.cuda.empty_cache()

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
        return self.runner._run_model(input_ids, positions, attn_meta, logits_indices)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Run a model forward pass on a scheduler batch (via ModelRunner)."""
        self._set_logits_gather(self._can_distribute_sampling(batch))
        return self.runner.forward(batch)

    def _can_distribute_sampling(self, batch: Batch) -> bool:
        """Return whether this batch can use bounded local-vocab sampling."""
        if self.tp_size <= 1 or self.server_args.tp_sampling_mode not in (
            "distributed",
            "distributed_packed",
            "adaptive",
        ):
            return False
        if self.server_args.tp_sampling_mode == "adaptive":
            return all(
                req.sampling_params.top_p >= 1.0
                and (
                    req.sampling_params.temperature <= 0.0
                    or req.sampling_params.top_k > 0
                )
                for req in batch.reqs
            )
        return all(
            # Bounded local candidates are sufficient for greedy/top-k, but
            # top-p needs the complete vocabulary to compute global cumulative
            # probabilities. Keep it on the full-logits path until a global
            # nucleus implementation exists.
            req.sampling_params.top_p >= 1.0
            and (
                req.sampling_params.temperature <= 0.0 or req.sampling_params.top_k > 0
            )
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
            if self.server_args.tp_sampling_mode == "adaptive" and all(
                param.temperature <= 0.0 for param in params
            ):
                tokens = self._sample_adaptive_greedy(logits)
            elif self.server_args.tp_sampling_mode in (
                "distributed_packed",
                "adaptive",
            ):
                tokens = self._sample_distributed_packed(logits, params)
            else:
                tokens = self._sample_distributed(logits, params)
        else:
            tokens = (
                self.sampler.sample_batch(logits, params)
                if get_tp_rank() == 0
                else None
            )
        if (
            distributed
            and self.server_args.tp_sampling_mode == "adaptive"
            and all(param.temperature <= 0.0 for param in params)
        ):
            return tokens
        token_tensor = torch.empty(
            len(batch.reqs), dtype=torch.long, device=self.device
        )
        if tokens is not None:
            token_tensor.copy_(
                torch.tensor(tokens, dtype=torch.long, device=self.device)
            )
        from minisgl.engine.collectives import broadcast

        broadcast(token_tensor, src=0)
        return token_tensor.tolist()

    def _sample_adaptive_greedy(self, logits: torch.Tensor) -> list[int] | None:
        """Find global greedy IDs using two tiny reductions instead of gathers."""
        from minisgl.engine.collectives import all_reduce

        local_values, local_ids = logits.max(dim=-1)
        global_values = all_reduce(local_values.clone(), op="max")
        vocab_per_rank = self.model_args.vocab_size // self.tp_size
        global_ids = local_ids + get_tp_rank() * vocab_per_rank
        invalid = torch.full_like(global_ids, torch.iinfo(torch.int64).max)
        global_ids = torch.where(local_values == global_values, global_ids, invalid)
        global_ids = all_reduce(global_ids, op="min")
        return global_ids.tolist()

    def _sample_distributed(
        self, logits: torch.Tensor, params: list
    ) -> list[int] | None:
        """Sample from globally bounded candidates without full-vocab gather."""
        from minisgl.engine.collectives import all_gather

        vocab_per_rank = self.model_args.vocab_size // self.tp_size
        max_k = max((max(1, p.top_k) if p.temperature > 0 else 1) for p in params)
        max_k = min(max_k, logits.shape[-1])
        values, indices = logits.topk(max_k, dim=-1)
        values, indices = self._stabilize_greedy_candidates(
            logits, values, indices, params
        )
        indices = indices + get_tp_rank() * vocab_per_rank

        # Candidate rows are gathered along the last dimension, so the result
        # is [batch, tp_size * k] and can be passed to the existing sampler.
        values = all_gather(values, dim=-1)
        indices = all_gather(indices, dim=-1)
        if get_tp_rank() != 0:
            return None
        if all(p.temperature <= 0.0 for p in params):
            return self._greedy_candidate_ids(values, indices)
        sampled = self.sampler.sample_batch(values, params)
        return (
            indices.gather(1, torch.tensor(sampled, device=self.device).view(-1, 1))
            .squeeze(1)
            .tolist()
        )

    @staticmethod
    def _greedy_candidate_ids(values: torch.Tensor, indices: torch.Tensor) -> list[int]:
        """Select greedy IDs with the same tie break as full-vocab argmax.

        Candidate tensors are ordered by TP rank, while ``Tensor.argmax`` on
        gathered full logits breaks equal values by the lowest vocabulary ID.
        Sorting by ID before argmax preserves that behavior for both regular
        and byte-packed candidate exchange.
        """
        order = indices.argsort(dim=-1, stable=True)
        sorted_values = values.gather(1, order)
        sorted_indices = indices.gather(1, order)
        selected = sorted_indices.gather(1, sorted_values.argmax(dim=-1, keepdim=True))
        return selected.squeeze(1).tolist()

    @staticmethod
    def _stabilize_greedy_candidates(
        logits: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
        params: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure each greedy row contains its stable local argmax.

        ``topk`` is not required to resolve equal values by the lowest index.
        That matters even when ``k == 1``: full-vocabulary ``argmax`` keeps the
        first maximum, while local ``topk`` may discard it before TP gathering.
        Put the stable local argmax in the first candidate for greedy rows.
        """
        greedy_rows = [p.temperature <= 0.0 for p in params]
        if not any(greedy_rows):
            return values, indices
        row_mask = torch.tensor(greedy_rows, dtype=torch.bool, device=logits.device)
        local_indices = logits.argmax(dim=-1)
        row_indices = torch.arange(logits.shape[0], device=logits.device)
        values = values.clone()
        indices = indices.clone()
        values[row_mask, 0] = logits[row_indices[row_mask], local_indices[row_mask]]
        indices[row_mask, 0] = local_indices[row_mask]
        return values, indices

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
            # Copy the byte ranges before changing dtype. The index range can
            # start at an unaligned offset when value_bytes is not a multiple
            # of index_dtype.itemsize (for example FP16 with k=1 followed by
            # int64 indices).
            # Flatten before copying so the new byte storage starts at offset
            # zero. ``contiguous()`` alone preserves the slice's storage offset
            # (for example 2 bytes for FP16 values), which is still invalid
            # when the byte tensor is reinterpreted as int64.
            value_bytes_view = chunk[..., :value_bytes].reshape(-1).clone()
            index_bytes_view = chunk[..., value_bytes:].reshape(-1).clone()
            values.append(value_bytes_view.view(value_dtype).view(-1, k))
            indices.append(index_bytes_view.view(index_dtype).view(-1, k))
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
        values, indices = self._stabilize_greedy_candidates(
            logits, values, indices, params
        )
        indices = indices + get_tp_rank() * vocab_per_rank
        # Token IDs are bounded by the model vocabulary and therefore fit in
        # int32 for supported models.  Keeping the index field narrow is the
        # actual payload reduction provided by the packed path: FP16 values
        # plus int32 IDs use 6 bytes per candidate instead of 10 bytes with
        # the default int64 top-k indices.
        if self.model_args.vocab_size > torch.iinfo(torch.int32).max:
            raise ValueError("Packed sampling requires a vocabulary <= int32 max")
        indices = indices.to(torch.int32)
        packet, value_bytes, index_bytes = self._pack_sampling_candidates(
            values, indices
        )
        # Gather rows rather than the byte dimension. This makes the rank
        # boundary explicit and avoids relying on the collective backend's
        # handling of a byte packet whose second dimension is later split into
        # heterogeneous value/index fields.
        batch_size = packet.shape[0]
        packet_width = packet.shape[1]
        packet = all_gather(packet, dim=0)
        packet = (
            packet.view(self.tp_size, batch_size, packet_width)
            .transpose(0, 1)
            .reshape(batch_size, self.tp_size * packet_width)
        )
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
        if all(p.temperature <= 0.0 for p in params):
            return self._greedy_candidate_ids(values, indices)
        sampled = self.sampler.sample_batch(values, params)
        return (
            indices.gather(1, torch.tensor(sampled, device=self.device).view(-1, 1))
            .squeeze(1)
            .tolist()
        )

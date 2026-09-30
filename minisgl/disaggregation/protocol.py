"""Small, versioned messages used by the synchronous PD runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass

VERSION = 1


@dataclass(frozen=True, slots=True)
class TransferKey:
    session: str
    uid: int
    room: int = 0

    def pack(self) -> str:
        return json.dumps(
            (VERSION, self.session, self.uid, self.room), separators=(",", ":")
        )

    @classmethod
    def unpack(cls, value: str) -> "TransferKey":
        version, session, uid, room = json.loads(value)
        if version != VERSION:
            raise ValueError(f"unsupported transfer protocol version: {version}")
        return cls(session, int(uid), int(room))


@dataclass(frozen=True, slots=True)
class TransferMeta:
    key: TransferKey
    shape: tuple[int, ...]
    dtype: str
    page_ids: tuple[int, ...]
    token_count: int

    def pack(self) -> bytes:
        return json.dumps(
            {
                "v": VERSION,
                "key": self.key.pack(),
                "shape": self.shape,
                "dtype": self.dtype,
                "pages": self.page_ids,
                "tokens": self.token_count,
            },
            separators=(",", ":"),
        ).encode()

    @classmethod
    def unpack(cls, value: bytes) -> "TransferMeta":
        obj = json.loads(value)
        if obj["v"] != VERSION:
            raise ValueError(f"unsupported transfer protocol version: {obj['v']}")
        return cls(
            TransferKey.unpack(obj["key"]),
            tuple(obj["shape"]),
            obj["dtype"],
            tuple(obj["pages"]),
            int(obj["tokens"]),
        )

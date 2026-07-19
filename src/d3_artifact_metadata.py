"""Payload-free member metadata readers for D3 NPZ and PyTorch ZIP artifacts."""

from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass, asdict
import io
from pathlib import Path
import pickle
import struct
from typing import Any, Callable
import zipfile


@dataclass(frozen=True)
class TensorDescriptor:
    storage_dtype: str
    storage_key: str
    storage_location: str
    storage_size: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool


@dataclass(frozen=True)
class StorageDescriptor:
    dtype: str
    key: str
    location: str
    size: int


class StorageType:
    def __init__(self, name: str) -> None:
        self.name = name


def _rebuild_tensor(
    storage: StorageDescriptor,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    *extra: object,
) -> TensorDescriptor:
    requires_grad = bool(extra[0]) if extra else False
    return TensorDescriptor(
        storage_dtype=storage.dtype,
        storage_key=storage.key,
        storage_location=storage.location,
        storage_size=storage.size,
        storage_offset=int(storage_offset),
        shape=tuple(int(value) for value in size),
        stride=tuple(int(value) for value in stride),
        requires_grad=requires_grad,
    )


def _rebuild_parameter(value: TensorDescriptor, requires_grad: bool, *extra: object) -> TensorDescriptor:
    del extra
    return TensorDescriptor(**{**asdict(value), "requires_grad": bool(requires_grad)})


class MetadataUnpickler(pickle.Unpickler):
    """Reject arbitrary globals and rebuild tensor descriptors without storage reads."""

    _ALLOWED: dict[tuple[str, str], Callable[..., object] | type] = {
        ("collections", "OrderedDict"): OrderedDict,
        ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
        ("torch._utils", "_rebuild_tensor_v3"): _rebuild_tensor,
        ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
        ("torch._utils", "_rebuild_parameter_with_state"): _rebuild_parameter,
    }

    def find_class(self, module: str, name: str) -> object:
        allowed = self._ALLOWED.get((module, name))
        if allowed is not None:
            return allowed
        if module == "torch" and (name.endswith("Storage") or name == "UntypedStorage"):
            return StorageType(name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, saved_id: object) -> StorageDescriptor:
        if not isinstance(saved_id, tuple) or len(saved_id) < 5 or saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        storage_type, key, location, size = saved_id[1:5]
        dtype = storage_type.name if isinstance(storage_type, StorageType) else str(storage_type)
        return StorageDescriptor(dtype=dtype, key=str(key), location=str(location), size=int(size))


def _data_pickle_name(names: list[str]) -> str:
    matches = [name for name in names if name.endswith("/data.pkl") or name == "data.pkl"]
    if len(matches) != 1:
        raise ValueError(f"expected one data.pkl member, found {matches}")
    return matches[0]


def inspect_torch_zip(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickle_name = _data_pickle_name(names)
        pickle_payload = archive.read(pickle_name)
        value = MetadataUnpickler(io.BytesIO(pickle_payload)).load()
        storage_members = [name for name in names if "/data/" in name]
        return {
            "container_format": "pytorch-zip",
            "pickle_member": pickle_name,
            "pickle_bytes_read": len(pickle_payload),
            "tensor_storage_payload_bytes_read": 0,
            "zip_members": [
                {
                    "name": item.filename,
                    "compressed_bytes": item.compress_size,
                    "uncompressed_bytes": item.file_size,
                    "crc32": f"{item.CRC:08x}",
                    "is_tensor_storage": item.filename in storage_members,
                }
                for item in archive.infolist()
            ],
            "structure": _jsonable(value),
        }


def _npy_header(handle: Any) -> dict[str, object]:
    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    major, minor = handle.read(2)
    length_bytes = handle.read(2 if major == 1 else 4)
    header_length = struct.unpack("<H" if major == 1 else "<I", length_bytes)[0]
    header_payload = handle.read(header_length)
    header = ast.literal_eval(header_payload.decode("latin1").strip())
    if set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError(f"unexpected NPY header fields: {sorted(header)}")
    return {
        "npy_version": f"{major}.{minor}",
        "header_bytes_read": 8 + len(length_bytes) + header_length,
        "dtype": header["descr"],
        "fortran_order": bool(header["fortran_order"]),
        "shape": list(header["shape"]),
    }


def inspect_npz(path: Path) -> dict[str, object]:
    members: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            with archive.open(item) as handle:
                header = _npy_header(handle)
            members.append({
                "name": item.filename[:-4] if item.filename.endswith(".npy") else item.filename,
                "zip_name": item.filename,
                "compressed_bytes": item.compress_size,
                "uncompressed_bytes": item.file_size,
                "crc32": f"{item.CRC:08x}",
                "array_payload_bytes_read": 0,
                **header,
            })
    return {"container_format": "npz", "members": members}


def _jsonable(value: object) -> object:
    if isinstance(value, TensorDescriptor):
        return {"kind": "tensor", **asdict(value)}
    if isinstance(value, StorageDescriptor):
        return {"kind": "storage", **asdict(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": "metadata_repr", "type": type(value).__name__, "repr": repr(value)}

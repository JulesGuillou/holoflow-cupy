from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


def jax_dtype(dtype: np.dtype | type | str) -> np.dtype:
    normalized = np.dtype(dtype)

    if normalized in {
        np.dtype("float64"),
        np.dtype("complex128"),
    }:
        jax.config.update("jax_enable_x64", True)

    mapping: dict[np.dtype, np.dtype] = {
        np.dtype("uint8"): jnp.uint8,
        np.dtype("int8"): jnp.int8,
        np.dtype("int16"): jnp.int16,
        np.dtype("int32"): jnp.int32,
        np.dtype("int64"): jnp.int64,
        np.dtype("float16"): jnp.float16,
        np.dtype("float32"): jnp.float32,
        np.dtype("float64"): jnp.float64,
        np.dtype("complex64"): jnp.complex64,
        np.dtype("complex128"): jnp.complex128,
    }

    try:
        return np.dtype(mapping[normalized])
    except KeyError as exc:
        raise TypeError(f"Unsupported JAX benchmark dtype: {normalized}.") from exc


__all__ = ["jax_dtype"]

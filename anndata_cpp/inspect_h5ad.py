from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


def shape_string(shape: Sequence[int]) -> str:
    if not shape:
        return "[]"
    return "[" + ", ".join(str(int(dim)) for dim in shape) + "]"


def numeric_dtype_name(dtype: Any) -> str:
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype)


def scalar_kind(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return "int64"
    if isinstance(value, (float, np.floating)):
        return "double"
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return "string"
    return type(value).__name__


def column_length(column: pd.Series | pd.Index) -> int:
    return len(column)


def column_kind(column: pd.Series | pd.Index) -> str:
    dtype = column.dtype
    if pd.api.types.is_categorical_dtype(dtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        return f"numeric({numeric_dtype_name(dtype)})"
    if pd.api.types.is_string_dtype(dtype) or dtype == object:
        return "string-array"
    return str(dtype)


def element_summary(value: Any) -> str:
    if sparse.issparse(value):
        return f"{value.getformat()} {shape_string(value.shape)} nnz={value.nnz}"
    if isinstance(value, pd.DataFrame):
        return f"dataframe rows={value.shape[0]} cols={value.shape[1]}"
    if isinstance(value, Mapping):
        return f"mapping keys={len(value)}"
    if isinstance(value, (pd.Series, pd.Index)):
        kind = column_kind(value)
        if kind.startswith("numeric("):
            return f"numeric-array {numeric_dtype_name(value.dtype)} {shape_string((len(value),))}"
        if kind == "categorical":
            categorical = pd.Categorical(value)
            return (
                f"categorical codes={shape_string(categorical.codes.shape)} "
                f"categories={len(categorical.categories)}"
            )
        return f"string-array {shape_string((len(value),))}"
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S", "O"}:
            return f"string-array {shape_string(value.shape)}"
        return f"numeric-array {numeric_dtype_name(value.dtype)} {shape_string(value.shape)}"
    if np.isscalar(value):
        return f"scalar {scalar_kind(value)}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return f"sequence len={len(value)}"
    return type(value).__name__


def print_mapping(name: str, mapping: Mapping[str, Any] | None) -> None:
    print(f"{name}: ", end="")
    if mapping is None:
        print("absent")
        return

    print(f"{len(mapping)} item(s)")
    for key, value in mapping.items():
        print(f"  - {key}: {element_summary(value)}")


def print_dataframe(name: str, dataframe: pd.DataFrame) -> None:
    print(
        f"{name}: rows={len(dataframe.index)} cols={len(dataframe.columns)} "
        f"index={dataframe.index.name} ({column_kind(dataframe.index)})"
    )
    for column_name in dataframe.columns:
        column = dataframe[column_name]
        print(f"  - {column_name}: {column_kind(column)} len={column_length(column)}")


def raw_items(raw: Any) -> dict[str, Any]:
    items: dict[str, Any] = {"X": raw.X, "var": raw.var}
    varm = getattr(raw, "varm", None)
    if varm is not None and len(varm) > 0:
        items["varm"] = dict(varm.items())
    return items


def print_summary(path: Path, adata: Any) -> None:
    print(f"file: {path}")

    print("X: ", end="")
    if adata.X is None:
        print("absent")
    else:
        print(element_summary(adata.X))
        print(adata.X)


    print_dataframe("obs", adata.obs)
    print("#################### OBS check:")
    print(adata.obs)
    
    print_dataframe("var", adata.var)
    print("#################### var check:")
    print(adata.var)
    print_mapping("obsm", dict(adata.obsm.items()))
    print("#################### obsm check:")
    print(dict(adata.obsm.items()))
    
    print_mapping("varm", dict(adata.varm.items()))
    print("#################### varm check:")
    print(dict(adata.varm.items()))
    
    print_mapping("obsp", dict(adata.obsp.items()))
    print("#################### obsp check:")
    print(dict(adata.obsp.items()))
    
    print_mapping("varp", dict(adata.varp.items()))
    
    print_mapping("layers", dict(adata.obsp.items()))
    print("#################### layers check:")
    print(dict(adata.layers.items()))

    print_mapping("layers", dict(adata.layers.items()))
    print_mapping("uns", dict(adata.uns.items()))
    
    print("#################### uns check:")
    print(dict(adata.uns.items()))
    
    print_mapping("raw", None if adata.raw is None else raw_items(adata.raw))


def load_anndata_module() -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    sys.path.insert(0, str(src_root))

    try:
        import anndata
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "a required dependency"
        raise SystemExit(
            "error: failed to import the local Python anndata reader because "
            f"`{missing_name}` is not installed.\n"
            "Install the Python dependencies from pyproject.toml, then run:\n"
            "  python3 anndata_cpp/inspect_h5ad.py <file.h5ad>"
        ) from exc

    return anndata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a .h5ad file using the Python anndata reader."
    )
    parser.add_argument("file", type=Path, help="Path to the .h5ad file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    anndata = load_anndata_module()

    try:
        adata = anndata.read_h5ad(args.file)
        print_summary(args.file, adata)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

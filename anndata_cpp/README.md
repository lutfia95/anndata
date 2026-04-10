# anndata_cpp

## Goal

Read the modern AnnData HDF5 layout directly from C++, with a data model that is intentionally close to the Python object graph:

- `AnnData`
- dense numeric arrays
- string arrays
- categorical columns
- CSR / CSC sparse matrices
- dataframes
- recursive mappings
- numeric and string scalars

The entry point is:

```cpp
anndata_cpp::AnnData adata = anndata_cpp::read_h5ad("file.h5ad");
```

## Current Support

- Root AnnData groups with `"encoding-type": "anndata"` and `"encoding-version": "0.1.0"`
- Dataframes with `"encoding-type": "dataframe"` and `"encoding-version": "0.2.0"`
- Dense arrays with `"encoding-type": "array"` and `"encoding-version": "0.2.0"`
- String arrays with `"encoding-type": "string-array"` and `"encoding-version": "0.2.0"`
- Categoricals with `"encoding-type": "categorical"` and `"encoding-version": "0.2.0"`
- Sparse matrices with `"csr_matrix"` / `"csc_matrix"` and `"encoding-version": "0.1.0"`
- Dict mappings with `"encoding-type": "dict"` and `"encoding-version": "0.1.0"`
- Scalars with `"numeric-scalar"` / `"string"` and `"encoding-version": "0.2.0"`
- `raw` groups represented as mappings

## Current Limits

- This build is `.h5ad` only. Zarr support has been removed.
- The implementation targets the modern tagged format, not the full historical compatibility surface of Python `anndata`.
- Older archives like `v0.7.x` are expected to fail for now.
- The code links directly against the installed HDF5 runtime (`libhdf5_serial`) because the development headers are not available in this environment.

## Build

```bash
cmake -S anndata_cpp -B anndata_cpp/build
cmake --build anndata_cpp/build
ctest --test-dir anndata_cpp/build --output-on-failure
```

## Run The C++ Inspector

From the repository root:

```bash
./anndata_cpp/build/anndata_cpp_main anndata_cpp/test_data/test.h5ad
```

Or from inside `anndata_cpp/`:

```bash
./build/anndata_cpp_main test_data/test.h5ad
```

The executable takes one argument:

```bash
anndata_cpp_main <file.h5ad>
```

## What `main.cpp` Prints

The current `main.cpp` is a small inspection tool for debugging and Python-parity checks. Right now it prints:

- the file path
- a summary of `X`
- a dataframe summary for `obs` and `var`
- the first four rows of `obs`
- the first four rows of `var`
- mapping summaries for `obsm`, `varm`, `obsp`, `varp`, `layers`, `uns`, and `raw`
- a 4-row preview of selected dense mappings using `print_matrix_head`

At the moment the matrix previews are hardcoded in `main.cpp` for:

- `obsm["X_umap"]`
- `varm["gene_stuff"]`

This is useful for checking a known file layout quickly. If your file uses different keys, update those names in `src/main.cpp`.

## Example Output

For the bundled test file, the inspector prints output like:

```text
file: "anndata_cpp/test_data/test.h5ad"
X: csr [100, 2000] nnz=126129
obs: rows=100 cols=1 index=_index (string-array)
  - cell_type: categorical len=100
  row 0 index=Cell_0 cell_type=T
  row 1 index=Cell_1 cell_type=Monocyte
var: rows=2000 cols=0 index=_index (string-array)
  row 0 index=Gene_0
obsm: 1 item(s)
  - X_umap: numeric-array float64 [100, 2]
```

The exact extra debug lines depend on what is enabled in `src/main.cpp`.

## Python Comparison

There is also a Python-side inspector that uses the local repo's `anndata.read_h5ad` path to print a matching summary:

```bash
python3 anndata_cpp/inspect_h5ad.py anndata_cpp/test_data/test.h5ad
```

That script requires the Python dependencies from `pyproject.toml`, especially `h5py`.

## Tests

The tests cover:

- synthetic modern `.h5ad` files written during the test run
- dense and sparse `X`
- dataframe indexes and columns
- categorical columns
- scalar values in `uns`
- the real archived fixture `tests/data/archives/v0.11.4/adata.h5ad`
- a negative compatibility test against `tests/data/archives/v0.7.8/adata.h5ad`

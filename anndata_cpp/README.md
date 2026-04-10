# anndata_cpp

`anndata_cpp` is now scoped only to `.h5ad` files.

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

## Quick Test

After building, point the small CLI at a file:

```bash
./anndata_cpp/build/anndata_cpp_main tests/data/archives/v0.11.4/adata.h5ad
```

It will load the file and print a compact summary of `X`, `obs`, `var`, and the mapping slots.

## Tests

The tests cover:

- synthetic modern `.h5ad` files written during the test run
- dense and sparse `X`
- dataframe indexes and columns
- categorical columns
- scalar values in `uns`
- the real archived fixture `tests/data/archives/v0.11.4/adata.h5ad`
- a negative compatibility test against `tests/data/archives/v0.7.8/adata.h5ad`

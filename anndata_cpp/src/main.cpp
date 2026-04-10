#include "anndata_cpp/anndata.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <variant>
#include <algorithm>
#include <cstdint>


namespace {

using anndata_cpp::AnnData;
using anndata_cpp::Categorical;
using anndata_cpp::Column;
using anndata_cpp::Element;
using anndata_cpp::Mapping;
using anndata_cpp::NumericArray;
using anndata_cpp::Scalar;
using anndata_cpp::SparseMatrix;
using anndata_cpp::StringArray;

/**
 * @brief Converts a local numeric dtype enum into a human-readable name.
 *
 * The CLI uses this helper when summarizing dense arrays, sparse payloads, and
 * dataframe column types in a compact textual form.
 *
 * @param dtype The numeric dtype enum value to describe.
 * @return A short string name for the dtype.
 */
std::string numeric_dtype_name(NumericArray::DType dtype) {
    switch (dtype) {
        case NumericArray::DType::kBool:
            return "bool";
        case NumericArray::DType::kUInt8:
            return "uint8";
        case NumericArray::DType::kUInt16:
            return "uint16";
        case NumericArray::DType::kUInt32:
            return "uint32";
        case NumericArray::DType::kUInt64:
            return "uint64";
        case NumericArray::DType::kInt8:
            return "int8";
        case NumericArray::DType::kInt16:
            return "int16";
        case NumericArray::DType::kInt32:
            return "int32";
        case NumericArray::DType::kInt64:
            return "int64";
        case NumericArray::DType::kFloat32:
            return "float32";
        case NumericArray::DType::kFloat64:
            return "float64";
    }
    return "unknown";
}


template <typename ShapeLike>
/**
 * @brief Formats a shape-like container as `[d0, d1, ...]`.
 *
 * @tparam ShapeLike Any container with `size()` and index access.
 * @param shape The dimensions to format.
 * @return A compact string representation of the shape.
 */
std::string shape_string(const ShapeLike& shape) {
    if (shape.empty()) {
        return "[]";
    }

    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

/**
 * @brief Returns the logical row count of a dataframe column variant.
 *
 * Numeric, string, and categorical column representations all store their
 * lengths differently, so the CLI normalizes them here for display purposes.
 *
 * @param column The dataframe column variant to inspect.
 * @return The number of logical entries stored in the column.
 */
std::size_t column_length(const Column& column) {
    return std::visit(
        [](const auto& value) -> std::size_t {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, NumericArray> || std::is_same_v<T, StringArray>) {
                if constexpr (std::is_same_v<T, NumericArray>) {
                    return value.element_count();
                } else {
                    return value.values.size();
                }
            } else if constexpr (std::is_same_v<T, Categorical>) {
                return value.codes.element_count();
            } else {
                return 0;
            }
        },
        column
    );
}

/**
 * @brief Returns a human-readable description of a dataframe column type.
 *
 * @param column The dataframe column variant to describe.
 * @return A short string describing the column encoding.
 */
std::string column_kind(const Column& column) {
    return std::visit(
        [](const auto& value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, NumericArray>) {
                return "numeric(" + numeric_dtype_name(value.dtype) + ")";
            } else if constexpr (std::is_same_v<T, StringArray>) {
                return "string-array";
            } else if constexpr (std::is_same_v<T, Categorical>) {
                return "categorical";
            } else {
                return "unknown";
            }
        },
        column
    );
}

/**
 * @brief Returns a human-readable description of a scalar variant type.
 *
 * @param scalar The scalar value to inspect.
 * @return A short string naming the scalar type.
 */
std::string scalar_kind(const Scalar& scalar) {
    if (scalar.is_bool()) {
        return "bool";
    }
    if (scalar.is_int()) {
        return "int64";
    }
    if (scalar.is_double()) {
        return "double";
    }
    if (scalar.is_string()) {
        return "string";
    }
    return "unknown";
}

/**
 * @brief Reads one integer-like value from a `NumericArray`.
 *
 * This helper is mainly used for categorical codes where the stored dtype may
 * vary across files but still needs to be interpreted as an integer index.
 *
 * @param array The numeric array to access.
 * @param i The element index to read.
 * @return The requested value converted to `std::int64_t`.
 */
std::int64_t numeric_int_at(const NumericArray& array, std::size_t i) {
    switch (array.dtype) {
        case NumericArray::DType::kBool:
            return array.values<std::uint8_t>().at(i) ? 1 : 0;
        case NumericArray::DType::kUInt8:
            return array.values<std::uint8_t>().at(i);
        case NumericArray::DType::kUInt16:
            return array.values<std::uint16_t>().at(i);
        case NumericArray::DType::kUInt32:
            return array.values<std::uint32_t>().at(i);
        case NumericArray::DType::kUInt64:
            return static_cast<std::int64_t>(array.values<std::uint64_t>().at(i));
        case NumericArray::DType::kInt8:
            return array.values<std::int8_t>().at(i);
        case NumericArray::DType::kInt16:
            return array.values<std::int16_t>().at(i);
        case NumericArray::DType::kInt32:
            return array.values<std::int32_t>().at(i);
        case NumericArray::DType::kInt64:
            return array.values<std::int64_t>().at(i);
        case NumericArray::DType::kFloat32:
        case NumericArray::DType::kFloat64:
            throw anndata_cpp::Error("numeric_int_at requires an integer-like dtype");
    }
    throw anndata_cpp::Error("numeric_int_at encountered an unknown dtype");
}

/**
 * @brief Formats one numeric array element as a string.
 *
 * The function dispatches on the stored dtype and materializes the requested
 * element using the matching C++ value type.
 *
 * @param array The numeric array to read from.
 * @param i The element index to format.
 * @return A string representation of the selected element.
 */
std::string numeric_value_as_string(const NumericArray& array, std::size_t i) {
    switch (array.dtype) {
        case NumericArray::DType::kBool:
            return array.values<std::uint8_t>().at(i) ? "true" : "false";
        case NumericArray::DType::kUInt8:
            return std::to_string(array.values<std::uint8_t>().at(i));
        case NumericArray::DType::kUInt16:
            return std::to_string(array.values<std::uint16_t>().at(i));
        case NumericArray::DType::kUInt32:
            return std::to_string(array.values<std::uint32_t>().at(i));
        case NumericArray::DType::kUInt64:
            return std::to_string(array.values<std::uint64_t>().at(i));
        case NumericArray::DType::kInt8:
            return std::to_string(array.values<std::int8_t>().at(i));
        case NumericArray::DType::kInt16:
            return std::to_string(array.values<std::int16_t>().at(i));
        case NumericArray::DType::kInt32:
            return std::to_string(array.values<std::int32_t>().at(i));
        case NumericArray::DType::kInt64:
            return std::to_string(array.values<std::int64_t>().at(i));
        case NumericArray::DType::kFloat32:
            return std::to_string(array.values<float>().at(i));
        case NumericArray::DType::kFloat64:
            return std::to_string(array.values<double>().at(i));
    }
    return "?";
}

/**
 * @brief Formats one dataframe column value as a string.
 *
 * The helper supports dense numeric columns, string columns, and categorical
 * columns so row previews can be printed uniformly.
 *
 * @param column The column variant to read from.
 * @param row The row index to format.
 * @return A string representation of the selected cell.
 */
std::string column_value_as_string(const Column& column, std::size_t row) {
    return std::visit(
        [row](const auto& value) -> std::string {
            using T = std::decay_t<decltype(value)>;

            if constexpr (std::is_same_v<T, NumericArray>) {
                return numeric_value_as_string(value, row);
            } else if constexpr (std::is_same_v<T, StringArray>) {
                return value.values.at(row);
            } else if constexpr (std::is_same_v<T, Categorical>) {
                const std::int64_t code = numeric_int_at(value.codes, row);
                if (code < 0 || static_cast<std::size_t>(code) >= value.categories.values.size()) {
                    return "<NA>";
                }
                return value.categories.values.at(static_cast<std::size_t>(code));
            } else {
                return "?";
            }
        },
        column
    );
}

/**
 * @brief Prints the first few rows of a decoded dataframe.
 *
 * The output includes the index value and all named columns for each printed
 * row, which makes it useful for quick debug comparisons with Python output.
 *
 * @param dataframe The dataframe to preview.
 * @param n_rows The maximum number of rows to print.
 */
void print_first_rows(const anndata_cpp::DataFrame& dataframe, std::size_t n_rows = 4) {
    const std::size_t n = std::min<std::size_t>(n_rows, column_length(dataframe.index));

    for (std::size_t row = 0; row < n; ++row) {
        std::cout << "  row " << row
                  << " index=" << column_value_as_string(dataframe.index, row);

        for (const auto& column_name : dataframe.column_order) {
            const auto it = dataframe.columns.find(column_name);
            if (it == dataframe.columns.end()) {
                continue;
            }
            std::cout << " " << column_name << "="
                      << column_value_as_string(it->second, row);
        }
        std::cout << '\n';
    }
}

/**
 * @brief Produces a compact one-line summary of a generic AnnData element.
 *
 * @param element The element to summarize.
 * @return A short description of the element kind, dtype, and shape.
 */
std::string element_summary(const Element& element) {
    if (element.is_numeric_array()) {
        const auto& array = element.as_numeric_array();
        return "numeric-array " + numeric_dtype_name(array.dtype) + " " + shape_string(array.shape);
    }
    if (element.is_string_array()) {
        const auto& array = element.as_string_array();
        return "string-array " + shape_string(array.shape);
    }
    if (element.is_categorical()) {
        const auto& categorical = element.as_categorical();
        return "categorical codes=" + shape_string(categorical.codes.shape) +
            " categories=" + std::to_string(categorical.categories.values.size());
    }
    if (element.is_sparse()) {
        const auto& sparse = element.as_sparse();
        return std::string(sparse.format == SparseMatrix::Format::kCsr ? "csr" : "csc") +
            " " + shape_string(std::vector<std::size_t>{sparse.shape.first, sparse.shape.second}) +
            " nnz=" + std::to_string(sparse.data.element_count());
    }
    if (element.is_dataframe()) {
        const auto& dataframe = element.as_dataframe();
        return "dataframe rows=" + std::to_string(column_length(dataframe.index)) +
            " cols=" + std::to_string(dataframe.column_order.size());
    }
    if (element.is_mapping()) {
        return "mapping keys=" + std::to_string(element.as_mapping().items.size());
    }
    if (element.is_scalar()) {
        return "scalar " + scalar_kind(element.as_scalar());
    }
    return "unknown";
}

/**
 * @brief Prints a summary of a mapping slot such as `obsm` or `layers`.
 *
 * Each child element is reported on its own line using `element_summary`.
 *
 * @param name The logical slot name being printed.
 * @param mapping The mapping pointer to inspect.
 */
void print_mapping(std::string_view name, const std::shared_ptr<Mapping>& mapping) {
    std::cout << name << ": ";
    if (mapping == nullptr) {
        std::cout << "absent\n";
        return;
    }

    std::cout << mapping->items.size() << " item(s)\n";
    for (const auto& [key, value] : mapping->items) {
        std::cout << "  - " << key << ": " << element_summary(value) << '\n';
    }
}

/**
 * @brief Prints a summary of a decoded dataframe slot.
 *
 * The current CLI also emits small row previews for `obs` and `var` so the
 * output is easier to compare against the Python-side inspector.
 *
 * @param name The dataframe slot name, such as `obs` or `var`.
 * @param dataframe The dataframe to summarize.
 */
void print_dataframe(std::string_view name, const anndata_cpp::DataFrame& dataframe) {
    std::cout << name << ": rows=" << column_length(dataframe.index)
              << " cols=" << dataframe.column_order.size()
              << " index=" << dataframe.index_name
              << " (" << column_kind(dataframe.index) << ")\n";
    if (!dataframe.column_order.empty()){
        std::cout << "First col: " << dataframe.column_order.front() << '\n';
    }

    for (const auto& column_name : dataframe.column_order) {
        const auto it = dataframe.columns.find(column_name);
        if (it == dataframe.columns.end()) {
            continue;
        }
        std::cout << "  - " << column_name << ": "
                  << column_kind(it->second)
                  << " len=" << column_length(it->second)
                  << '\n';
    }
    if (name == "obs"){
        std::cout << "################ OBS Check" << std::endl;
        print_first_rows(dataframe, 4);
    }
    if (name == "var"){
        std::cout << "################ var Check" << std::endl;
        print_first_rows(dataframe, 4);
    }


}

using MappingPtr = std::shared_ptr<anndata_cpp::Mapping>;

/**
 * @brief Prints the leading rows of a matrix-like mapping element.
 *
 * Dense numeric arrays are printed directly. CSR sparse matrices are expanded
 * row-by-row into dense temporary rows so the first few rows can be inspected
 * in a Python-like debugging workflow.
 *
 * @param mapping The mapping that owns the keyed element.
 * @param key The child key to print, such as `X_umap`.
 * @param max_rows The maximum number of leading rows to print.
 */
void print_matrix_head(
      const MappingPtr& mapping,
      const std::string& key,
      std::size_t max_rows = 4
  ) {
      if (!mapping) {
          std::cout << "mapping is null\n";
          return;
      }

      const auto it = mapping->items.find(key);
      if (it == mapping->items.end()) {
          std::cout << "key not found: " << key << '\n';
          return;
      }

      const auto& elem = it->second;

      if (elem.is_numeric_array()) {
          const auto& arr = elem.as_numeric_array();
          if (arr.shape.size() != 2) {
              std::cout << "array is not 2D: " << key << '\n';
              return;
          }

          const std::size_t rows = arr.shape[0];
          const std::size_t cols = arr.shape[1];
          const std::size_t n = std::min(max_rows, rows);

          switch (arr.dtype) {
              case anndata_cpp::NumericArray::DType::kFloat32: {
                  auto values = arr.values<float>();
                  for (std::size_t r = 0; r < n; ++r) {
                      for (std::size_t c = 0; c < cols; ++c) {
                          std::cout << values[r * cols + c] << ' ';
                      }
                      std::cout << '\n';
                  }
                  break;
              }
              case anndata_cpp::NumericArray::DType::kFloat64: {
                  auto values = arr.values<double>();
                  for (std::size_t r = 0; r < n; ++r) {
                      for (std::size_t c = 0; c < cols; ++c) {
                          std::cout << values[r * cols + c] << ' ';
                      }
                      std::cout << '\n';
                  }
                  break;
              }
              default:
                  std::cout << "unsupported dense dtype for " << key << '\n';
                  break;
          }
          return;
      }

      if (elem.is_sparse()) {
          const auto& sp = elem.as_sparse();
          const std::size_t rows = sp.shape.first;
          const std::size_t cols = sp.shape.second;
          const std::size_t n = std::min(max_rows, rows);

          if (sp.format != anndata_cpp::SparseMatrix::Format::kCsr) {
              std::cout << "only CSR sparse printing implemented for " << key << '\n';
              return;
          }

          auto indices = sp.indices.values<std::int32_t>();
          auto indptr = sp.indptr.values<std::int32_t>();

          if (sp.data.dtype == anndata_cpp::NumericArray::DType::kFloat32) {
              auto data = sp.data.values<float>();

              for (std::size_t r = 0; r < n; ++r) {
                  std::vector<float> row(cols, 0.0f);
                  for (std::int32_t p = indptr[r]; p < indptr[r + 1]; ++p) {
                      row[indices[p]] = data[p];
                  }
                  for (std::size_t c = 0; c < cols; ++c) {
                      std::cout << row[c] << ' ';
                  }
                  std::cout << '\n';
              }
              return;
          }

          if (sp.data.dtype == anndata_cpp::NumericArray::DType::kFloat64) {
              auto data = sp.data.values<double>();

              for (std::size_t r = 0; r < n; ++r) {
                  std::vector<double> row(cols, 0.0);
                  for (std::int32_t p = indptr[r]; p < indptr[r + 1]; ++p) {
                      row[indices[p]] = data[p];
                  }
                  for (std::size_t c = 0; c < cols; ++c) {
                      std::cout << row[c] << ' ';
                  }
                  std::cout << '\n';
              }
              return;
          }

          std::cout << "unsupported sparse dtype for " << key << '\n';
          return;
      }

	      std::cout << "element is neither dense nor sparse matrix: " << key << '\n';
	  }

/**
 * @brief Prints the current high-level inspection summary for an AnnData file.
 *
 * In addition to the standard slot summaries, the current debug executable also
 * emits a few hard-coded previews for known keys used by the bundled fixture.
 *
 * @param path The file path that was loaded.
 * @param adata The decoded AnnData object to inspect.
 */
void print_summary(const std::filesystem::path& path, const AnnData& adata) {
    std::cout << "file: " << path << '\n';

    std::cout << "X: ";
    if (!adata.X.has_value()) {
        std::cout << "absent\n";
    } else {
        std::cout << element_summary(*adata.X) << '\n';
        const auto& x = adata.X->as_sparse();

        switch (x.data.dtype) {
            case anndata_cpp::NumericArray::DType::kFloat32: {
                auto data = x.data.values<float>();
                std::cout << data[3] << '\n';
                break;
            }
            case anndata_cpp::NumericArray::DType::kFloat64: {
                auto data = x.data.values<double>();
                std::cout << data[3] << '\n';
                break;
            }
            default:
                std::cout << "unexpected dtype\n";
                break;
        }

        //auto data = x.data.values<int>();      // or float/int depending on dtype
        auto indices = x.indices.values<int32_t>();
        auto indptr = x.indptr.values<int32_t>();
    }

    print_dataframe("obs", adata.obs);
    print_dataframe("var", adata.var);

    print_mapping("obsm", adata.obsm);
    std::cout << "OBSM check #####" << '\n';
    print_matrix_head(adata.obsm, "X_umap", 4);
    
    print_mapping("varm", adata.varm);
    std::cout << "varm check #####" << '\n';
    print_matrix_head(adata.varm, "gene_stuff", 4);

    print_mapping("obsp", adata.obsp);
    print_mapping("varp", adata.varp);

    print_mapping("layers", adata.layers);
    std::cout << "layers check #####" << '\n';
    //print_matrix_head(adata.layers, "log_transformed", 4);

    print_mapping("uns", adata.uns);
	    print_mapping("raw", adata.raw);
}

} 

/**
 * @brief Entry point for the small `.h5ad` inspection executable.
 *
 * The program expects a single file path, reads it through `read_h5ad`, and
 * prints the current debug summary to standard output.
 *
 * @param argc The number of command-line arguments.
 * @param argv The raw command-line argument vector.
 * @return `EXIT_SUCCESS` on success, otherwise `EXIT_FAILURE`.
 */
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: anndata_cpp_main <file.h5ad>\n";
        return EXIT_FAILURE;
    }

    try {
        const std::filesystem::path path = argv[1];
        const AnnData adata = anndata_cpp::read_h5ad(path);
        print_summary(path, adata);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

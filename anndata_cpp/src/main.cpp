#include "anndata_cpp/anndata.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <variant>

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

void print_dataframe(std::string_view name, const anndata_cpp::DataFrame& dataframe) {
    std::cout << name << ": rows=" << column_length(dataframe.index)
              << " cols=" << dataframe.column_order.size()
              << " index=" << dataframe.index_name
              << " (" << column_kind(dataframe.index) << ")\n";

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
}

void print_summary(const std::filesystem::path& path, const AnnData& adata) {
    std::cout << "file: " << path << '\n';

    std::cout << "X: ";
    if (!adata.X.has_value()) {
        std::cout << "absent\n";
    } else {
        std::cout << element_summary(*adata.X) << '\n';
    }

    print_dataframe("obs", adata.obs);
    print_dataframe("var", adata.var);
    print_mapping("obsm", adata.obsm);
    print_mapping("varm", adata.varm);
    print_mapping("obsp", adata.obsp);
    print_mapping("varp", adata.varp);
    print_mapping("layers", adata.layers);
    print_mapping("uns", adata.uns);
    print_mapping("raw", adata.raw);
}

}  // namespace

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

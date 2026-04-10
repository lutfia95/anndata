#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace anndata_cpp {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct NumericArray {
    enum class DType {
        kBool,
        kUInt8,
        kUInt16,
        kUInt32,
        kUInt64,
        kInt8,
        kInt16,
        kInt32,
        kInt64,
        kFloat32,
        kFloat64,
    };

    std::vector<std::size_t> shape;
    DType dtype = DType::kInt32;
    std::vector<std::uint8_t> bytes;

    std::size_t element_count() const;
    std::size_t item_size() const;

    template <typename T>
    std::vector<T> values() const {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
        if (bytes.size() % sizeof(T) != 0) {
            throw Error("numeric array byte size does not match requested type");
        }
        std::vector<T> out(bytes.size() / sizeof(T));
        if (!bytes.empty()) {
            std::memcpy(out.data(), bytes.data(), bytes.size());
        }
        return out;
    }
};

struct StringArray {
    std::vector<std::size_t> shape;
    std::vector<std::string> values;
};

struct Categorical {
    NumericArray codes;
    StringArray categories;
    bool ordered = false;
};

struct SparseMatrix {
    enum class Format {
        kCsr,
        kCsc,
    };

    Format format = Format::kCsr;
    std::pair<std::size_t, std::size_t> shape = {0, 0};
    NumericArray data;
    NumericArray indices;
    NumericArray indptr;
};

struct Scalar {
    using Value = std::variant<bool, std::int64_t, double, std::string>;

    Value value;

    bool is_bool() const;
    bool is_int() const;
    bool is_double() const;
    bool is_string() const;

    bool as_bool() const;
    std::int64_t as_int() const;
    double as_double() const;
    const std::string& as_string() const;
};

using Column = std::variant<NumericArray, StringArray, Categorical>;

struct DataFrame {
    std::string index_name;
    Column index;
    std::vector<std::string> column_order;
    std::map<std::string, Column> columns;
};

struct Mapping;

struct Element {
    using Value = std::variant<
        NumericArray,
        StringArray,
        Categorical,
        SparseMatrix,
        DataFrame,
        std::shared_ptr<Mapping>,
        Scalar
    >;

    Value value;

    bool is_numeric_array() const;
    bool is_string_array() const;
    bool is_categorical() const;
    bool is_sparse() const;
    bool is_dataframe() const;
    bool is_mapping() const;
    bool is_scalar() const;

    const NumericArray& as_numeric_array() const;
    const StringArray& as_string_array() const;
    const Categorical& as_categorical() const;
    const SparseMatrix& as_sparse() const;
    const DataFrame& as_dataframe() const;
    const Mapping& as_mapping() const;
    const Scalar& as_scalar() const;
};

struct Mapping {
    std::map<std::string, Element> items;
};

struct AnnData {
    std::optional<Element> X;
    DataFrame obs;
    DataFrame var;
    std::shared_ptr<Mapping> obsm;
    std::shared_ptr<Mapping> varm;
    std::shared_ptr<Mapping> obsp;
    std::shared_ptr<Mapping> varp;
    std::shared_ptr<Mapping> layers;
    std::shared_ptr<Mapping> uns;
    std::shared_ptr<Mapping> raw;
};

AnnData read_h5ad(const std::filesystem::path& path);

}  // namespace anndata_cpp

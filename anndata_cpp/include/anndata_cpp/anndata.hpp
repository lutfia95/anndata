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

/**
 * @brief Represents parser-level errors raised while decoding an `.h5ad` file.
 *
 * This exception type is used throughout the C++ reader to report unsupported
 * encodings, malformed HDF5 layouts, and type mismatches encountered while
 * constructing the in-memory AnnData model.
 */
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Stores a dense numeric array together with its logical shape and dtype.
 *
 * The payload is stored as a raw byte buffer so the reader can preserve the
 * original HDF5 numeric dtype and only materialize typed values on demand.
 */
struct NumericArray {
    /**
     * @brief Enumerates the numeric dtypes supported by the reader.
     *
     * These values mirror the dense and sparse numeric element types the
     * parser can recognize in modern `.h5ad` files.
     */
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

    /**
     * @brief Returns the logical number of elements described by `shape`.
     *
     * The result is the product of all dimensions in the array shape. It is
     * used by the CLI helpers and sparse summaries to report element counts.
     *
     * @return The total element count implied by the stored shape.
     */
    std::size_t element_count() const;

    /**
     * @brief Returns the byte width of a single element for the stored dtype.
     *
     * This method maps the array dtype to the number of bytes required for one
     * value, which is useful when validating decoded dataset buffers.
     *
     * @return The size in bytes of one array element.
     */
    std::size_t item_size() const;

    template <typename T>
    /**
     * @brief Materializes the raw byte buffer as a typed vector.
     *
     * This method copies the stored bytes into a `std::vector<T>` after
     * verifying that the byte count is compatible with the requested type.
     * It does not perform numeric conversion; callers must request the exact
     * C++ type that matches the stored dtype.
     *
     * @tparam T The target trivially copyable value type.
     * @return A typed copy of the array contents.
     * @throws Error If the byte size is incompatible with `T`.
     */
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

/**
 * @brief Stores a dense string array together with its logical shape.
 *
 * String payloads are decoded eagerly into a vector of C++ strings while the
 * original multidimensional shape is preserved separately.
 */
struct StringArray {
    std::vector<std::size_t> shape;
    std::vector<std::string> values;
};

/**
 * @brief Represents a categorical array as integer codes plus string categories.
 *
 * This mirrors the tagged AnnData categorical encoding where codes are stored
 * separately from the category labels and the ordered flag.
 */
struct Categorical {
    NumericArray codes;
    StringArray categories;
    bool ordered = false;
};

/**
 * @brief Stores a sparse matrix in CSR or CSC form.
 *
 * The matrix keeps its format tag, two-dimensional shape, and the three
 * constituent arrays required to reconstruct the sparse structure.
 */
struct SparseMatrix {
    /**
     * @brief Identifies the sparse storage layout used by the matrix.
     *
     * Modern AnnData files can store sparse matrices in compressed sparse row
     * or compressed sparse column form.
     */
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

/**
 * @brief Holds a scalar value decoded from an AnnData scalar element.
 *
 * Scalars are normalized into one of the supported primitive value types so
 * callers can inspect them without dealing with raw HDF5 APIs.
 */
struct Scalar {
    using Value = std::variant<bool, std::int64_t, double, std::string>;

    Value value;

    /**
     * @brief Checks whether the stored scalar value is boolean.
     *
     * @return `true` when the scalar holds a boolean value.
     */
    bool is_bool() const;

    /**
     * @brief Checks whether the stored scalar value is a signed integer.
     *
     * @return `true` when the scalar holds an `std::int64_t`.
     */
    bool is_int() const;

    /**
     * @brief Checks whether the stored scalar value is floating point.
     *
     * @return `true` when the scalar holds a `double`.
     */
    bool is_double() const;

    /**
     * @brief Checks whether the stored scalar value is string-like.
     *
     * @return `true` when the scalar holds a `std::string`.
     */
    bool is_string() const;

    /**
     * @brief Returns the stored boolean scalar value.
     *
     * @return The contained boolean value.
     */
    bool as_bool() const;

    /**
     * @brief Returns the stored integer scalar value.
     *
     * @return The contained signed integer value.
     */
    std::int64_t as_int() const;

    /**
     * @brief Returns the stored floating-point scalar value.
     *
     * @return The contained double value.
     */
    double as_double() const;

    /**
     * @brief Returns the stored string scalar value.
     *
     * @return A const reference to the contained string.
     */
    const std::string& as_string() const;
};

/**
 * @brief Represents a dataframe column decoded from `obs` or `var`.
 *
 * AnnData dataframe columns may be dense numeric arrays, string arrays, or
 * categorical values, so a variant is used to preserve the original encoding.
 */
using Column = std::variant<NumericArray, StringArray, Categorical>;

/**
 * @brief Models an AnnData dataframe group such as `obs` or `var`.
 *
 * The dataframe stores the logical index column, column ordering, and the
 * decoded column payloads keyed by column name.
 */
struct DataFrame {
    std::string index_name;
    Column index;
    std::vector<std::string> column_order;
    std::map<std::string, Column> columns;
};

struct Mapping;

/**
 * @brief Wraps any supported AnnData element in a single variant container.
 *
 * This type is used for recursive mappings such as `obsm`, `layers`, and
 * `uns`, where each key may point to a different encoded element type.
 */
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

    /**
     * @brief Checks whether the element stores a dense numeric array.
     *
     * @return `true` when the active variant alternative is `NumericArray`.
     */
    bool is_numeric_array() const;

    /**
     * @brief Checks whether the element stores a dense string array.
     *
     * @return `true` when the active variant alternative is `StringArray`.
     */
    bool is_string_array() const;

    /**
     * @brief Checks whether the element stores a categorical array.
     *
     * @return `true` when the active variant alternative is `Categorical`.
     */
    bool is_categorical() const;

    /**
     * @brief Checks whether the element stores a sparse matrix.
     *
     * @return `true` when the active variant alternative is `SparseMatrix`.
     */
    bool is_sparse() const;

    /**
     * @brief Checks whether the element stores a dataframe.
     *
     * @return `true` when the active variant alternative is `DataFrame`.
     */
    bool is_dataframe() const;

    /**
     * @brief Checks whether the element stores a recursive mapping.
     *
     * @return `true` when the active variant alternative is `std::shared_ptr<Mapping>`.
     */
    bool is_mapping() const;

    /**
     * @brief Checks whether the element stores a scalar value.
     *
     * @return `true` when the active variant alternative is `Scalar`.
     */
    bool is_scalar() const;

    /**
     * @brief Returns the element as a dense numeric array.
     *
     * @return A const reference to the stored `NumericArray`.
     */
    const NumericArray& as_numeric_array() const;

    /**
     * @brief Returns the element as a dense string array.
     *
     * @return A const reference to the stored `StringArray`.
     */
    const StringArray& as_string_array() const;

    /**
     * @brief Returns the element as a categorical array.
     *
     * @return A const reference to the stored `Categorical`.
     */
    const Categorical& as_categorical() const;

    /**
     * @brief Returns the element as a sparse matrix.
     *
     * @return A const reference to the stored `SparseMatrix`.
     */
    const SparseMatrix& as_sparse() const;

    /**
     * @brief Returns the element as a dataframe.
     *
     * @return A const reference to the stored `DataFrame`.
     */
    const DataFrame& as_dataframe() const;

    /**
     * @brief Returns the element as a recursive mapping.
     *
     * @return A const reference to the stored `Mapping`.
     */
    const Mapping& as_mapping() const;

    /**
     * @brief Returns the element as a scalar.
     *
     * @return A const reference to the stored `Scalar`.
     */
    const Scalar& as_scalar() const;
};

/**
 * @brief Stores a dictionary-like AnnData group.
 *
 * Mapping groups such as `obsm`, `varm`, `layers`, `uns`, and `raw` are
 * represented as key-value collections of recursively typed `Element`s.
 */
struct Mapping {
    std::map<std::string, Element> items;
};

/**
 * @brief Represents the top-level AnnData object decoded from an `.h5ad` file.
 *
 * The structure is intentionally close to the Python object graph so callers
 * can inspect the same major slots from C++ with minimal translation.
 */
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

/**
 * @brief Reads a modern `.h5ad` file into the C++ AnnData model.
 *
 * This function opens the HDF5 file, validates the tagged AnnData root group,
 * and recursively decodes the supported element types into an `AnnData`
 * instance.
 *
 * @param path The filesystem path to the `.h5ad` file to parse.
 * @return The fully decoded AnnData object.
 * @throws Error If the file is malformed or uses an unsupported encoding.
 */
AnnData read_h5ad(const std::filesystem::path& path);

}  // namespace anndata_cpp

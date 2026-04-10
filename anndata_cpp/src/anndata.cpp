#include "anndata_cpp/anndata.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "hdf5_api.hpp"

namespace anndata_cpp {
namespace {

using h5::hid_t;

/**
 * @brief Owns an HDF5 identifier together with the function needed to close it.
 *
 * This small RAII wrapper keeps file, group, dataset, datatype, dataspace, and
 * attribute handles exception-safe by automatically releasing them when they go
 * out of scope.
 */
class Handle {
public:
    using CloseFn = h5::herr_t (*)(hid_t);

    Handle() = default;

    Handle(hid_t id, CloseFn close_fn) : id_(id), close_fn_(close_fn) {}

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    Handle(Handle&& other) noexcept : id_(other.id_), close_fn_(other.close_fn_) {
        other.id_ = -1;
        other.close_fn_ = nullptr;
    }

    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            reset();
            id_ = other.id_;
            close_fn_ = other.close_fn_;
            other.id_ = -1;
            other.close_fn_ = nullptr;
        }
        return *this;
    }

    ~Handle() {
        reset();
    }

    hid_t get() const {
        return id_;
    }

    explicit operator bool() const {
        return id_ >= 0;
    }

private:
    void reset() {
        if (id_ >= 0 && close_fn_ != nullptr) {
            close_fn_(id_);
        }
        id_ = -1;
        close_fn_ = nullptr;
    }

    hid_t id_ = -1;
    CloseFn close_fn_ = nullptr;
};

/**
 * @brief Concatenates an HDF5 parent path and child name into one absolute path.
 *
 * The helper keeps root handling consistent so recursive readers can build
 * paths like `/obs/cell_type` without duplicating slash logic everywhere.
 *
 * @param base The existing absolute parent path.
 * @param leaf The child path component to append.
 * @return The combined absolute HDF5 path.
 */
std::string join_path(std::string_view base, std::string_view leaf) {
    if (base.empty() || base == "/") {
        return "/" + std::string(leaf);
    }
    return std::string(base) + "/" + std::string(leaf);
}

/**
 * @brief Throws an `Error` when an HDF5 call reports failure.
 *
 * @param status The status code returned by an HDF5 function.
 * @param message The parser error message to raise on failure.
 * @throws Error If `status` is negative.
 */
void expect_ok(h5::herr_t status, const std::string& message) {
    if (status < 0) {
        throw Error(message);
    }
}

/**
 * @brief Throws an `Error` when an HDF5 identifier is invalid.
 *
 * @param id The HDF5 identifier returned by an open or create call.
 * @param message The parser error message to raise on failure.
 * @throws Error If `id` is negative.
 */
void expect_valid(hid_t id, const std::string& message) {
    if (id < 0) {
        throw Error(message);
    }
}

/**
 * @brief Opens an `.h5ad` file in read-only mode.
 *
 * The function also ensures the HDF5 runtime is initialized before attempting
 * to open the file.
 *
 * @param path The filesystem path to the target `.h5ad` file.
 * @return An owning handle for the opened HDF5 file.
 */
Handle open_file_readonly(const std::filesystem::path& path) {
    h5::initialize();
    const hid_t file_id = h5::H5Fopen(path.c_str(), h5::kFileAccRdOnly, h5::kDefault);
    expect_valid(file_id, "failed to open h5ad file: " + path.string());
    return Handle(file_id, &h5::H5Fclose);
}

/**
 * @brief Opens an arbitrary HDF5 object by absolute path.
 *
 * @param loc_id The parent location from which the path is resolved.
 * @param path The absolute HDF5 object path to open.
 * @return An owning handle for the opened object.
 */
Handle open_object(hid_t loc_id, const std::string& path) {
    const hid_t object_id = h5::H5Oopen(loc_id, path.c_str(), h5::kDefault);
    expect_valid(object_id, "failed to open HDF5 object at " + path);
    return Handle(object_id, &h5::H5Oclose);
}

/**
 * @brief Opens an HDF5 group by path.
 *
 * @param loc_id The parent location from which the group is resolved.
 * @param path The absolute group path to open.
 * @return An owning handle for the opened group.
 */
Handle open_group(hid_t loc_id, const std::string& path) {
    const hid_t group_id = h5::H5Gopen2(loc_id, path.c_str(), h5::kDefault);
    expect_valid(group_id, "failed to open HDF5 group at " + path);
    return Handle(group_id, &h5::H5Gclose);
}

/**
 * @brief Opens an HDF5 dataset by path.
 *
 * @param loc_id The parent location from which the dataset is resolved.
 * @param path The absolute dataset path to open.
 * @return An owning handle for the opened dataset.
 */
Handle open_dataset(hid_t loc_id, const std::string& path) {
    const hid_t dataset_id = h5::H5Dopen2(loc_id, path.c_str(), h5::kDefault);
    expect_valid(dataset_id, "failed to open HDF5 dataset at " + path);
    return Handle(dataset_id, &h5::H5Dclose);
}

/**
 * @brief Opens a required attribute on an HDF5 object.
 *
 * @param obj_id The object that owns the attribute.
 * @param name The attribute name to open.
 * @param context A human-readable description of the object for error messages.
 * @return An owning handle for the opened attribute.
 * @throws Error If the attribute does not exist or cannot be opened.
 */
Handle open_attribute(hid_t obj_id, const std::string& name, const std::string& context) {
    if (h5::H5Aexists(obj_id, name.c_str()) <= 0) {
        throw Error(context + " is missing required attribute '" + name + "'");
    }
    const hid_t attr_id = h5::H5Aopen_name(obj_id, name.c_str());
    expect_valid(attr_id, "failed to open attribute '" + name + "' on " + context);
    return Handle(attr_id, &h5::H5Aclose);
}

/**
 * @brief Retrieves the datatype associated with a dataset.
 *
 * @param dataset_id The dataset identifier.
 * @param context A human-readable description used in error messages.
 * @return An owning handle for the dataset datatype.
 */
Handle get_dataset_type(hid_t dataset_id, const std::string& context) {
    const hid_t type_id = h5::H5Dget_type(dataset_id);
    expect_valid(type_id, "failed to read datatype for " + context);
    return Handle(type_id, &h5::H5Tclose);
}

/**
 * @brief Retrieves the dataspace associated with a dataset.
 *
 * @param dataset_id The dataset identifier.
 * @param context A human-readable description used in error messages.
 * @return An owning handle for the dataset dataspace.
 */
Handle get_dataset_space(hid_t dataset_id, const std::string& context) {
    const hid_t space_id = h5::H5Dget_space(dataset_id);
    expect_valid(space_id, "failed to read dataspace for " + context);
    return Handle(space_id, &h5::H5Sclose);
}

/**
 * @brief Retrieves the datatype associated with an attribute.
 *
 * @param attr_id The attribute identifier.
 * @param context A human-readable description used in error messages.
 * @return An owning handle for the attribute datatype.
 */
Handle get_attribute_type(hid_t attr_id, const std::string& context) {
    const hid_t type_id = h5::H5Aget_type(attr_id);
    expect_valid(type_id, "failed to read attribute datatype for " + context);
    return Handle(type_id, &h5::H5Tclose);
}

/**
 * @brief Retrieves the dataspace associated with an attribute.
 *
 * @param attr_id The attribute identifier.
 * @param context A human-readable description used in error messages.
 * @return An owning handle for the attribute dataspace.
 */
Handle get_attribute_space(hid_t attr_id, const std::string& context) {
    const hid_t space_id = h5::H5Aget_space(attr_id);
    expect_valid(space_id, "failed to read attribute dataspace for " + context);
    return Handle(space_id, &h5::H5Sclose);
}

/**
 * @brief Converts an HDF5 dataspace into a vector of extents.
 *
 * Scalar dataspaces are represented as an empty shape vector, while array
 * dataspaces are converted into `std::size_t` dimensions.
 *
 * @param space_id The dataspace identifier to inspect.
 * @return The decoded logical shape.
 */
std::vector<std::size_t> get_shape(hid_t space_id) {
    const int ndims = h5::H5Sget_simple_extent_ndims(space_id);
    if (ndims < 0) {
        throw Error("failed to query dataspace rank");
    }
    if (ndims == 0) {
        return {};
    }
    std::vector<h5::hsize_t> dims(static_cast<std::size_t>(ndims));
    expect_ok(
        h5::H5Sget_simple_extent_dims(space_id, dims.data(), nullptr),
        "failed to query dataspace dimensions"
    );
    std::vector<std::size_t> out;
    out.reserve(dims.size());
    for (const auto dim : dims) {
        out.push_back(static_cast<std::size_t>(dim));
    }
    return out;
}

/**
 * @brief Computes the product of a shape vector.
 *
 * The empty-shape case is treated as one element, which matches the scalar
 * semantics used throughout the parser.
 *
 * @param shape The shape dimensions to multiply.
 * @return The total number of elements implied by `shape`.
 */
std::size_t product(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(
        shape.begin(),
        shape.end(),
        static_cast<std::size_t>(1),
        [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; }
    );
}

/**
 * @brief Trims a fixed-width C string at the first null terminator.
 *
 * @param bytes The start of the fixed-width character buffer.
 * @param width The number of bytes available in the buffer.
 * @return The decoded C++ string without trailing padding bytes.
 */
std::string trim_c_string(const char* bytes, std::size_t width) {
    std::size_t length = 0;
    while (length < width && bytes[length] != '\0') {
        ++length;
    }
    return std::string(bytes, length);
}

/**
 * @brief Checks whether an HDF5 link exists at the given path.
 *
 * @param loc_id The parent location from which the path is resolved.
 * @param path The absolute HDF5 path to test.
 * @return `true` when the link exists.
 */
bool link_exists(hid_t loc_id, const std::string& path) {
    return h5::H5Lexists(loc_id, path.c_str(), h5::kDefault) > 0;
}

/**
 * @brief Returns the sorted child names of an HDF5 group.
 *
 * The names are sorted to keep mapping iteration deterministic across runs.
 *
 * @param loc_id The parent location from which the group is resolved.
 * @param path The absolute group path to enumerate.
 * @return A sorted vector of child names.
 */
std::vector<std::string> group_children(hid_t loc_id, const std::string& path) {
    const Handle group = open_group(loc_id, path);
    h5::hsize_t num_objects = 0;
    expect_ok(h5::H5Gget_num_objs(group.get(), &num_objects), "failed to enumerate " + path);

    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(num_objects));
    for (h5::hsize_t idx = 0; idx < num_objects; ++idx) {
        const long name_len = h5::H5Gget_objname_by_idx(group.get(), idx, nullptr, 0);
        if (name_len < 0) {
            throw Error("failed to read child name in " + path);
        }
        std::string name(static_cast<std::size_t>(name_len), '\0');
        const long copied = h5::H5Gget_objname_by_idx(
            group.get(),
            idx,
            name.data(),
            name.size() + 1
        );
        if (copied < 0) {
            throw Error("failed to read child name in " + path);
        }
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

/**
 * @brief Reads a scalar string attribute from an HDF5 object.
 *
 * Both fixed-width and variable-length UTF-8 string attributes are supported.
 *
 * @param obj_id The object that owns the attribute.
 * @param name The attribute name to read.
 * @param context A human-readable description used in error messages.
 * @return The decoded string value.
 */
std::string read_string_attribute(hid_t obj_id, const std::string& name, const std::string& context) {
    const Handle attr = open_attribute(obj_id, name, context);
    const Handle type = get_attribute_type(attr.get(), context);
    const Handle space = get_attribute_space(attr.get(), context);
    const auto shape = get_shape(space.get());
    if (!shape.empty()) {
        throw Error("attribute '" + name + "' on " + context + " is not scalar");
    }
    if (h5::H5Tget_class(type.get()) != h5::TClass::kString) {
        throw Error("attribute '" + name + "' on " + context + " is not a string");
    }

    if (h5::H5Tis_variable_str(type.get()) > 0) {
        Handle mem_type(h5::H5Tcopy(h5::H5T_C_S1_g), &h5::H5Tclose);
        expect_valid(mem_type.get(), "failed to copy string datatype");
        expect_ok(h5::H5Tset_size(mem_type.get(), h5::kVariable), "failed to set variable string size");
        expect_ok(h5::H5Tset_cset(mem_type.get(), h5::kCsetUtf8), "failed to set UTF-8 string charset");
        expect_ok(h5::H5Tset_strpad(mem_type.get(), h5::kStrNullterm), "failed to set string padding");

        char* value = nullptr;
        expect_ok(h5::H5Aread(attr.get(), mem_type.get(), &value), "failed to read string attribute");
        const std::string out = value == nullptr ? std::string() : std::string(value);
        std::vector<char*> reclaim = {value};
        expect_ok(h5::H5Dvlen_reclaim(mem_type.get(), space.get(), h5::kDefault, reclaim.data()), "failed to reclaim string attribute");
        return out;
    }

    const std::size_t width = h5::H5Tget_size(type.get());
    std::vector<char> bytes(width);
    expect_ok(h5::H5Aread(attr.get(), type.get(), bytes.data()), "failed to read fixed string attribute");
    return trim_c_string(bytes.data(), width);
}

/**
 * @brief Reads a string-vector attribute from an HDF5 object.
 *
 * This helper is used for metadata such as dataframe column order where the
 * attribute stores a collection of strings rather than a scalar.
 *
 * @param obj_id The object that owns the attribute.
 * @param name The attribute name to read.
 * @param context A human-readable description used in error messages.
 * @return The decoded string vector.
 */
std::vector<std::string> read_string_vector_attribute(
    hid_t obj_id,
    const std::string& name,
    const std::string& context
) {
    const Handle attr = open_attribute(obj_id, name, context);
    const Handle type = get_attribute_type(attr.get(), context);
    const Handle space = get_attribute_space(attr.get(), context);
    const auto shape = get_shape(space.get());
    const std::size_t count = product(shape);
    std::vector<std::string> out;
    out.reserve(count);

    if (count == 0) {
        return out;
    }

    if (h5::H5Tget_class(type.get()) != h5::TClass::kString) {
        throw Error("attribute '" + name + "' on " + context + " is not a string array");
    }

    if (h5::H5Tis_variable_str(type.get()) > 0) {
        Handle mem_type(h5::H5Tcopy(h5::H5T_C_S1_g), &h5::H5Tclose);
        expect_valid(mem_type.get(), "failed to copy string datatype");
        expect_ok(h5::H5Tset_size(mem_type.get(), h5::kVariable), "failed to set variable string size");
        expect_ok(h5::H5Tset_cset(mem_type.get(), h5::kCsetUtf8), "failed to set UTF-8 string charset");
        expect_ok(h5::H5Tset_strpad(mem_type.get(), h5::kStrNullterm), "failed to set string padding");

        std::vector<char*> values(count, nullptr);
        expect_ok(h5::H5Aread(attr.get(), mem_type.get(), values.data()), "failed to read string vector attribute");
        for (char* value : values) {
            out.emplace_back(value == nullptr ? "" : value);
        }
        expect_ok(h5::H5Dvlen_reclaim(mem_type.get(), space.get(), h5::kDefault, values.data()), "failed to reclaim string vector attribute");
        return out;
    }

    const std::size_t width = h5::H5Tget_size(type.get());
    std::vector<char> bytes(count * width);
    expect_ok(h5::H5Aread(attr.get(), type.get(), bytes.data()), "failed to read fixed string vector attribute");
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(trim_c_string(bytes.data() + (i * width), width));
    }
    return out;
}

/**
 * @brief Reads an integer-like attribute into signed 64-bit values.
 *
 * Integer and enum HDF5 attributes are both normalized into `std::int64_t`
 * values so callers can use the same helper for shapes and boolean flags.
 *
 * @param obj_id The object that owns the attribute.
 * @param name The attribute name to read.
 * @param context A human-readable description used in error messages.
 * @return The decoded integer values.
 */
std::vector<std::int64_t> read_int_vector_attribute(
    hid_t obj_id,
    const std::string& name,
    const std::string& context
) {
    const Handle attr = open_attribute(obj_id, name, context);
    const Handle type = get_attribute_type(attr.get(), context);
    const Handle space = get_attribute_space(attr.get(), context);
    const auto shape = get_shape(space.get());
    const std::size_t count = product(shape);

    const h5::TClass type_class = h5::H5Tget_class(type.get());
    if (type_class != h5::TClass::kInteger && type_class != h5::TClass::kEnum) {
        throw Error("attribute '" + name + "' on " + context + " is not integer-like");
    }

    std::vector<std::int64_t> out(count, 0);
    expect_ok(h5::H5Aread(attr.get(), h5::H5T_NATIVE_INT64_g, out.data()), "failed to read integer vector attribute");
    return out;
}

/**
 * @brief Reads a boolean attribute encoded as an integer-like scalar.
 *
 * @param obj_id The object that owns the attribute.
 * @param name The attribute name to read.
 * @param context A human-readable description used in error messages.
 * @return The decoded boolean flag.
 */
bool read_bool_attribute(hid_t obj_id, const std::string& name, const std::string& context) {
    const auto values = read_int_vector_attribute(obj_id, name, context);
    if (values.size() != 1) {
        throw Error("attribute '" + name + "' on " + context + " is not scalar");
    }
    return values[0] != 0;
}

/**
 * @brief Maps an HDF5 numeric datatype to the local `NumericArray::DType`.
 *
 * @param type_id The HDF5 datatype identifier to inspect.
 * @param context A human-readable description used in error messages.
 * @return The matching local numeric dtype.
 */
NumericArray::DType map_numeric_dtype(hid_t type_id, const std::string& context) {
    const h5::TClass type_class = h5::H5Tget_class(type_id);
    const std::size_t size = h5::H5Tget_size(type_id);

    if (type_class == h5::TClass::kFloat) {
        if (size == 4) {
            return NumericArray::DType::kFloat32;
        }
        if (size == 8) {
            return NumericArray::DType::kFloat64;
        }
    }

    if (type_class == h5::TClass::kEnum && size == 1) {
        return NumericArray::DType::kBool;
    }

    if (type_class == h5::TClass::kInteger || type_class == h5::TClass::kEnum) {
        const h5::TSign sign = h5::H5Tget_sign(type_id);
        if (sign == h5::TSign::kNone) {
            if (size == 1) {
                return NumericArray::DType::kUInt8;
            }
            if (size == 2) {
                return NumericArray::DType::kUInt16;
            }
            if (size == 4) {
                return NumericArray::DType::kUInt32;
            }
            if (size == 8) {
                return NumericArray::DType::kUInt64;
            }
        } else {
            if (size == 1) {
                return NumericArray::DType::kInt8;
            }
            if (size == 2) {
                return NumericArray::DType::kInt16;
            }
            if (size == 4) {
                return NumericArray::DType::kInt32;
            }
            if (size == 8) {
                return NumericArray::DType::kInt64;
            }
        }
    }

    throw Error("unsupported HDF5 numeric datatype at " + context);
}

/**
 * @brief Returns the native HDF5 memory type corresponding to a local dtype.
 *
 * @param dtype The local numeric dtype.
 * @return The matching HDF5 native memory datatype identifier.
 */
hid_t native_type_for(NumericArray::DType dtype) {
    switch (dtype) {
        case NumericArray::DType::kBool:
            return h5::H5T_NATIVE_UCHAR_g;
        case NumericArray::DType::kUInt8:
            return h5::H5T_NATIVE_UINT8_g;
        case NumericArray::DType::kUInt16:
            return h5::H5T_NATIVE_UINT16_g;
        case NumericArray::DType::kUInt32:
            return h5::H5T_NATIVE_UINT32_g;
        case NumericArray::DType::kUInt64:
            return h5::H5T_NATIVE_UINT64_g;
        case NumericArray::DType::kInt8:
            return h5::H5T_NATIVE_INT8_g;
        case NumericArray::DType::kInt16:
            return h5::H5T_NATIVE_INT16_g;
        case NumericArray::DType::kInt32:
            return h5::H5T_NATIVE_INT32_g;
        case NumericArray::DType::kInt64:
            return h5::H5T_NATIVE_INT64_g;
        case NumericArray::DType::kFloat32:
            return h5::H5T_NATIVE_FLOAT_g;
        case NumericArray::DType::kFloat64:
            return h5::H5T_NATIVE_DOUBLE_g;
    }
    throw Error("unsupported numeric dtype");
}

/**
 * @brief Returns the byte width associated with a local numeric dtype.
 *
 * @param dtype The local numeric dtype.
 * @return The size in bytes of one value of that dtype.
 */
std::size_t item_size(NumericArray::DType dtype) {
    switch (dtype) {
        case NumericArray::DType::kBool:
        case NumericArray::DType::kUInt8:
        case NumericArray::DType::kInt8:
            return 1;
        case NumericArray::DType::kUInt16:
        case NumericArray::DType::kInt16:
            return 2;
        case NumericArray::DType::kUInt32:
        case NumericArray::DType::kInt32:
        case NumericArray::DType::kFloat32:
            return 4;
        case NumericArray::DType::kUInt64:
        case NumericArray::DType::kInt64:
        case NumericArray::DType::kFloat64:
            return 8;
    }
    throw Error("unsupported numeric dtype");
}

/**
 * @brief Reads a dense numeric dataset into a `NumericArray`.
 *
 * The dataset shape, dtype, and raw byte payload are all preserved so callers
 * can materialize typed values later using the exact stored dtype.
 *
 * @param loc_id The parent location from which the dataset path is resolved.
 * @param path The absolute dataset path to read.
 * @return The decoded dense numeric array.
 */
NumericArray read_numeric_dataset(hid_t loc_id, const std::string& path) {
    const Handle dataset = open_dataset(loc_id, path);
    const Handle type = get_dataset_type(dataset.get(), path);
    const Handle space = get_dataset_space(dataset.get(), path);

    NumericArray out;
    out.shape = get_shape(space.get());
    out.dtype = map_numeric_dtype(type.get(), path);
    out.bytes.resize(product(out.shape) * item_size(out.dtype));
    if (!out.bytes.empty()) {
        expect_ok(
            h5::H5Dread(
                dataset.get(),
                native_type_for(out.dtype),
                h5::kAll,
                h5::kAll,
                h5::kDefault,
                out.bytes.data()
            ),
            "failed to read numeric dataset at " + path
        );
    }
    return out;
}

/**
 * @brief Reads a dense string dataset into a `StringArray`.
 *
 * Both fixed-width and variable-length string datasets are supported.
 *
 * @param loc_id The parent location from which the dataset path is resolved.
 * @param path The absolute dataset path to read.
 * @return The decoded dense string array.
 */
StringArray read_string_dataset(hid_t loc_id, const std::string& path) {
    const Handle dataset = open_dataset(loc_id, path);
    const Handle type = get_dataset_type(dataset.get(), path);
    const Handle space = get_dataset_space(dataset.get(), path);

    if (h5::H5Tget_class(type.get()) != h5::TClass::kString) {
        throw Error("dataset at " + path + " is not a string dataset");
    }

    StringArray out;
    out.shape = get_shape(space.get());
    const std::size_t count = product(out.shape);
    out.values.reserve(count);

    if (count == 0) {
        return out;
    }

    if (h5::H5Tis_variable_str(type.get()) > 0) {
        Handle mem_type(h5::H5Tcopy(h5::H5T_C_S1_g), &h5::H5Tclose);
        expect_valid(mem_type.get(), "failed to copy string datatype");
        expect_ok(h5::H5Tset_size(mem_type.get(), h5::kVariable), "failed to set variable string size");
        expect_ok(h5::H5Tset_cset(mem_type.get(), h5::kCsetUtf8), "failed to set UTF-8 string charset");
        expect_ok(h5::H5Tset_strpad(mem_type.get(), h5::kStrNullterm), "failed to set string padding");

        std::vector<char*> values(count, nullptr);
        expect_ok(
            h5::H5Dread(dataset.get(), mem_type.get(), h5::kAll, h5::kAll, h5::kDefault, values.data()),
            "failed to read variable-length string dataset at " + path
        );
        for (char* value : values) {
            out.values.emplace_back(value == nullptr ? "" : value);
        }
        expect_ok(h5::H5Dvlen_reclaim(mem_type.get(), space.get(), h5::kDefault, values.data()), "failed to reclaim variable-length strings at " + path);
        return out;
    }

    const std::size_t width = h5::H5Tget_size(type.get());
    std::vector<char> bytes(count * width);
    expect_ok(
        h5::H5Dread(dataset.get(), type.get(), h5::kAll, h5::kAll, h5::kDefault, bytes.data()),
        "failed to read fixed string dataset at " + path
    );
    for (std::size_t i = 0; i < count; ++i) {
        out.values.push_back(trim_c_string(bytes.data() + (i * width), width));
    }
    return out;
}

Column read_column(hid_t loc_id, const std::string& path);
Element read_element(hid_t loc_id, const std::string& path);

/**
 * @brief Reads a tagged AnnData dataframe group such as `obs` or `var`.
 *
 * The reader validates the dataframe encoding metadata, resolves the stored
 * index column, and decodes each named dataframe column in column-order order.
 *
 * @param loc_id The parent location from which the dataframe path is resolved.
 * @param path The absolute dataframe group path to decode.
 * @return The decoded dataframe representation.
 */
DataFrame read_dataframe(hid_t loc_id, const std::string& path) {
    const Handle group = open_group(loc_id, path);
    const std::string version = read_string_attribute(group.get(), "encoding-version", path);
    if (read_string_attribute(group.get(), "encoding-type", path) != "dataframe") {
        throw Error("dataframe encoding-type mismatch at " + path);
    }
    if (version != "0.2.0") {
        throw Error("unsupported dataframe encoding-version at " + path + ": " + version);
    }

    DataFrame out;
    out.index_name = read_string_attribute(group.get(), "_index", path);
    out.column_order = read_string_vector_attribute(group.get(), "column-order", path);
    out.index = read_column(loc_id, join_path(path, out.index_name));
    for (const auto& column_name : out.column_order) {
        out.columns.emplace(column_name, read_column(loc_id, join_path(path, column_name)));
    }
    return out;
}

/**
 * @brief Reads a categorical column group.
 *
 * The categorical payload is reconstructed from its codes array, categories
 * string array, and ordered flag.
 *
 * @param loc_id The parent location from which the group path is resolved.
 * @param path The absolute categorical group path to decode.
 * @return The decoded categorical column.
 */
Categorical read_categorical(hid_t loc_id, const std::string& path) {
    const Handle group = open_group(loc_id, path);
    if (read_string_attribute(group.get(), "encoding-type", path) != "categorical") {
        throw Error("categorical encoding-type mismatch at " + path);
    }
    const std::string version = read_string_attribute(group.get(), "encoding-version", path);
    if (version != "0.2.0") {
        throw Error("unsupported categorical encoding-version at " + path + ": " + version);
    }

    Categorical out;
    out.ordered = read_bool_attribute(group.get(), "ordered", path);
    out.codes = read_numeric_dataset(loc_id, join_path(path, "codes"));
    out.categories = read_string_dataset(loc_id, join_path(path, "categories"));
    return out;
}

/**
 * @brief Reads a sparse matrix group in CSR or CSC form.
 *
 * The helper validates the encoding metadata, reads the declared matrix shape,
 * and decodes the `data`, `indices`, and `indptr` arrays needed to reconstruct
 * the sparse structure.
 *
 * @param loc_id The parent location from which the group path is resolved.
 * @param path The absolute sparse group path to decode.
 * @param format The expected sparse storage layout.
 * @return The decoded sparse matrix.
 */
SparseMatrix read_sparse(hid_t loc_id, const std::string& path, SparseMatrix::Format format) {
    const Handle group = open_group(loc_id, path);
    const std::string expected_type = format == SparseMatrix::Format::kCsr ? "csr_matrix" : "csc_matrix";
    if (read_string_attribute(group.get(), "encoding-type", path) != expected_type) {
        throw Error("sparse matrix encoding-type mismatch at " + path);
    }
    const std::string version = read_string_attribute(group.get(), "encoding-version", path);
    if (version != "0.1.0") {
        throw Error("unsupported sparse encoding-version at " + path + ": " + version);
    }

    const auto shape_values = read_int_vector_attribute(group.get(), "shape", path);
    if (shape_values.size() != 2) {
        throw Error("sparse matrix shape must have length 2 at " + path);
    }

    SparseMatrix out;
    out.format = format;
    out.shape = {
        static_cast<std::size_t>(shape_values[0]),
        static_cast<std::size_t>(shape_values[1]),
    };
    out.data = read_numeric_dataset(loc_id, join_path(path, "data"));
    out.indices = read_numeric_dataset(loc_id, join_path(path, "indices"));
    out.indptr = read_numeric_dataset(loc_id, join_path(path, "indptr"));
    return out;
}

/**
 * @brief Reads a tagged scalar dataset.
 *
 * Numeric and string scalar encodings are supported and normalized into the
 * local `Scalar` variant.
 *
 * @param loc_id The parent location from which the dataset path is resolved.
 * @param path The absolute scalar dataset path to decode.
 * @return The decoded scalar value.
 */
Scalar read_scalar(hid_t loc_id, const std::string& path) {
    const Handle dataset = open_dataset(loc_id, path);
    const Handle type = get_dataset_type(dataset.get(), path);
    const Handle space = get_dataset_space(dataset.get(), path);
    if (!get_shape(space.get()).empty()) {
        throw Error("scalar dataset at " + path + " is not rank-0");
    }

    const Handle object = open_object(loc_id, path);
    const std::string encoding_type = read_string_attribute(object.get(), "encoding-type", path);
    const std::string version = read_string_attribute(object.get(), "encoding-version", path);
    if (version != "0.2.0") {
        throw Error("unsupported scalar encoding-version at " + path + ": " + version);
    }

    if (encoding_type == "string") {
        const auto array = read_string_dataset(loc_id, path);
        if (array.values.size() != 1) {
            throw Error("string scalar at " + path + " did not decode to one value");
        }
        return Scalar{array.values[0]};
    }

    if (encoding_type != "numeric-scalar") {
        throw Error("unsupported scalar encoding-type at " + path + ": " + encoding_type);
    }

    const NumericArray::DType dtype = map_numeric_dtype(type.get(), path);
    switch (dtype) {
        case NumericArray::DType::kBool: {
            std::uint8_t value = 0;
            expect_ok(h5::H5Dread(dataset.get(), h5::H5T_NATIVE_UCHAR_g, h5::kAll, h5::kAll, h5::kDefault, &value), "failed to read bool scalar at " + path);
            return Scalar{value != 0};
        }
        case NumericArray::DType::kFloat32: {
            float value = 0.0F;
            expect_ok(h5::H5Dread(dataset.get(), h5::H5T_NATIVE_FLOAT_g, h5::kAll, h5::kAll, h5::kDefault, &value), "failed to read float scalar at " + path);
            return Scalar{static_cast<double>(value)};
        }
        case NumericArray::DType::kFloat64: {
            double value = 0.0;
            expect_ok(h5::H5Dread(dataset.get(), h5::H5T_NATIVE_DOUBLE_g, h5::kAll, h5::kAll, h5::kDefault, &value), "failed to read double scalar at " + path);
            return Scalar{value};
        }
        default: {
            std::int64_t value = 0;
            expect_ok(h5::H5Dread(dataset.get(), h5::H5T_NATIVE_INT64_g, h5::kAll, h5::kAll, h5::kDefault, &value), "failed to read integer scalar at " + path);
            return Scalar{value};
        }
    }
}

/**
 * @brief Reads a dictionary-like mapping group.
 *
 * Child names are enumerated deterministically and each element is decoded
 * recursively using the generic element reader.
 *
 * @param loc_id The parent location from which the group path is resolved.
 * @param path The absolute mapping group path to decode.
 * @param expected_type The required tagged mapping type, such as `dict` or `raw`.
 * @return The decoded recursive mapping.
 */
std::shared_ptr<Mapping> read_mapping(hid_t loc_id, const std::string& path, std::string_view expected_type) {
    const Handle group = open_group(loc_id, path);
    const std::string encoding_type = read_string_attribute(group.get(), "encoding-type", path);
    if (encoding_type != expected_type) {
        throw Error("mapping encoding-type mismatch at " + path + ": " + encoding_type);
    }
    const std::string version = read_string_attribute(group.get(), "encoding-version", path);
    if (version != "0.1.0") {
        throw Error("unsupported mapping encoding-version at " + path + ": " + version);
    }

    auto mapping = std::make_shared<Mapping>();
    for (const auto& child_name : group_children(loc_id, path)) {
        mapping->items.emplace(child_name, read_element(loc_id, join_path(path, child_name)));
    }
    return mapping;
}

/**
 * @brief Reads one dataframe column from a dataset or tagged group.
 *
 * Columns may be stored directly as dense datasets or as grouped categorical
 * elements, so this helper dispatches based on object kind and encoding tags.
 *
 * @param loc_id The parent location from which the column path is resolved.
 * @param path The absolute column path to decode.
 * @return The decoded dataframe column variant.
 */
Column read_column(hid_t loc_id, const std::string& path) {
    const Handle object = open_object(loc_id, path);
    if (h5::H5Iget_type(object.get()) == h5::IType::kDataset) {
        if (h5::H5Aexists(object.get(), "encoding-type") > 0) {
            const std::string encoding_type = read_string_attribute(object.get(), "encoding-type", path);
            if (encoding_type == "string-array") {
                return read_string_dataset(loc_id, path);
            }
            if (encoding_type == "array") {
                return read_numeric_dataset(loc_id, path);
            }
        }

        const Handle dataset = open_dataset(loc_id, path);
        const Handle type = get_dataset_type(dataset.get(), path);
        if (h5::H5Tget_class(type.get()) == h5::TClass::kString) {
            return read_string_dataset(loc_id, path);
        }
        return read_numeric_dataset(loc_id, path);
    }

    if (h5::H5Aexists(object.get(), "encoding-type") <= 0) {
        throw Error("column group at " + path + " is missing encoding-type");
    }
    const std::string encoding_type = read_string_attribute(object.get(), "encoding-type", path);
    if (encoding_type == "categorical") {
        return read_categorical(loc_id, path);
    }
    throw Error("unsupported dataframe column encoding-type at " + path + ": " + encoding_type);
}

/**
 * @brief Reads any supported AnnData element from a dataset or group path.
 *
 * This is the central dispatch function for recursive mappings and the top-
 * level `X` slot. It examines the HDF5 object kind and the tagged encoding
 * metadata to choose the correct specialized reader.
 *
 * @param loc_id The parent location from which the element path is resolved.
 * @param path The absolute element path to decode.
 * @return The decoded generic element wrapper.
 */
Element read_element(hid_t loc_id, const std::string& path) {
    const Handle object = open_object(loc_id, path);
    const h5::IType object_type = h5::H5Iget_type(object.get());

    if (object_type == h5::IType::kDataset) {
        if (h5::H5Aexists(object.get(), "encoding-type") > 0) {
            const std::string encoding_type = read_string_attribute(object.get(), "encoding-type", path);
            if (encoding_type == "array") {
                return Element{read_numeric_dataset(loc_id, path)};
            }
            if (encoding_type == "string-array") {
                return Element{read_string_dataset(loc_id, path)};
            }
            if (encoding_type == "numeric-scalar" || encoding_type == "string") {
                return Element{read_scalar(loc_id, path)};
            }
            throw Error("unsupported dataset encoding-type at " + path + ": " + encoding_type);
        }

        const Handle dataset = open_dataset(loc_id, path);
        const Handle type = get_dataset_type(dataset.get(), path);
        if (h5::H5Tget_class(type.get()) == h5::TClass::kString) {
            return Element{read_string_dataset(loc_id, path)};
        }
        return Element{read_numeric_dataset(loc_id, path)};
    }

    if (object_type != h5::IType::kGroup) {
        throw Error("unsupported HDF5 object at " + path);
    }

    if (h5::H5Aexists(object.get(), "encoding-type") <= 0) {
        throw Error("group at " + path + " is missing encoding-type");
    }

    const std::string encoding_type = read_string_attribute(object.get(), "encoding-type", path);
    if (encoding_type == "dataframe") {
        return Element{read_dataframe(loc_id, path)};
    }
    if (encoding_type == "dict") {
        return Element{read_mapping(loc_id, path, "dict")};
    }
    if (encoding_type == "raw") {
        return Element{read_mapping(loc_id, path, "raw")};
    }
    if (encoding_type == "csr_matrix") {
        return Element{read_sparse(loc_id, path, SparseMatrix::Format::kCsr)};
    }
    if (encoding_type == "csc_matrix") {
        return Element{read_sparse(loc_id, path, SparseMatrix::Format::kCsc)};
    }
    if (encoding_type == "categorical") {
        return Element{read_categorical(loc_id, path)};
    }

    throw Error("unsupported group encoding-type at " + path + ": " + encoding_type);
}

}  // namespace

/**
 * @brief Returns the number of elements represented by this numeric array.
 *
 * @return The product of the array dimensions.
 */
std::size_t NumericArray::element_count() const {
    return product(shape);
}

/**
 * @brief Returns the byte width of one stored numeric value.
 *
 * @return The number of bytes used by the array dtype.
 */
std::size_t NumericArray::item_size() const {
    return anndata_cpp::item_size(dtype);
}

/**
 * @brief Checks whether the scalar stores a boolean value.
 *
 * @return `true` when the active scalar variant alternative is `bool`.
 */
bool Scalar::is_bool() const {
    return std::holds_alternative<bool>(value);
}

/**
 * @brief Checks whether the scalar stores an integer value.
 *
 * @return `true` when the active scalar variant alternative is `std::int64_t`.
 */
bool Scalar::is_int() const {
    return std::holds_alternative<std::int64_t>(value);
}

/**
 * @brief Checks whether the scalar stores a floating-point value.
 *
 * @return `true` when the active scalar variant alternative is `double`.
 */
bool Scalar::is_double() const {
    return std::holds_alternative<double>(value);
}

/**
 * @brief Checks whether the scalar stores a string value.
 *
 * @return `true` when the active scalar variant alternative is `std::string`.
 */
bool Scalar::is_string() const {
    return std::holds_alternative<std::string>(value);
}

/**
 * @brief Returns the stored boolean scalar.
 *
 * @return The contained boolean value.
 */
bool Scalar::as_bool() const {
    return std::get<bool>(value);
}

/**
 * @brief Returns the stored integer scalar.
 *
 * @return The contained signed integer value.
 */
std::int64_t Scalar::as_int() const {
    return std::get<std::int64_t>(value);
}

/**
 * @brief Returns the stored floating-point scalar.
 *
 * @return The contained double value.
 */
double Scalar::as_double() const {
    return std::get<double>(value);
}

/**
 * @brief Returns the stored string scalar.
 *
 * @return A const reference to the contained string.
 */
const std::string& Scalar::as_string() const {
    return std::get<std::string>(value);
}

/**
 * @brief Checks whether this element stores a dense numeric array.
 *
 * @return `true` when the active element variant alternative is `NumericArray`.
 */
bool Element::is_numeric_array() const {
    return std::holds_alternative<NumericArray>(value);
}

/**
 * @brief Checks whether this element stores a dense string array.
 *
 * @return `true` when the active element variant alternative is `StringArray`.
 */
bool Element::is_string_array() const {
    return std::holds_alternative<StringArray>(value);
}

/**
 * @brief Checks whether this element stores a categorical array.
 *
 * @return `true` when the active element variant alternative is `Categorical`.
 */
bool Element::is_categorical() const {
    return std::holds_alternative<Categorical>(value);
}

/**
 * @brief Checks whether this element stores a sparse matrix.
 *
 * @return `true` when the active element variant alternative is `SparseMatrix`.
 */
bool Element::is_sparse() const {
    return std::holds_alternative<SparseMatrix>(value);
}

/**
 * @brief Checks whether this element stores a dataframe.
 *
 * @return `true` when the active element variant alternative is `DataFrame`.
 */
bool Element::is_dataframe() const {
    return std::holds_alternative<DataFrame>(value);
}

/**
 * @brief Checks whether this element stores a mapping.
 *
 * @return `true` when the active element variant alternative is a mapping pointer.
 */
bool Element::is_mapping() const {
    return std::holds_alternative<std::shared_ptr<Mapping>>(value);
}

/**
 * @brief Checks whether this element stores a scalar value.
 *
 * @return `true` when the active element variant alternative is `Scalar`.
 */
bool Element::is_scalar() const {
    return std::holds_alternative<Scalar>(value);
}

/**
 * @brief Returns this element as a dense numeric array.
 *
 * @return A const reference to the stored numeric array.
 */
const NumericArray& Element::as_numeric_array() const {
    return std::get<NumericArray>(value);
}

/**
 * @brief Returns this element as a dense string array.
 *
 * @return A const reference to the stored string array.
 */
const StringArray& Element::as_string_array() const {
    return std::get<StringArray>(value);
}

/**
 * @brief Returns this element as a categorical array.
 *
 * @return A const reference to the stored categorical array.
 */
const Categorical& Element::as_categorical() const {
    return std::get<Categorical>(value);
}

/**
 * @brief Returns this element as a sparse matrix.
 *
 * @return A const reference to the stored sparse matrix.
 */
const SparseMatrix& Element::as_sparse() const {
    return std::get<SparseMatrix>(value);
}

/**
 * @brief Returns this element as a dataframe.
 *
 * @return A const reference to the stored dataframe.
 */
const DataFrame& Element::as_dataframe() const {
    return std::get<DataFrame>(value);
}

/**
 * @brief Returns this element as a mapping.
 *
 * @return A const reference to the stored mapping.
 */
const Mapping& Element::as_mapping() const {
    return *std::get<std::shared_ptr<Mapping>>(value);
}

/**
 * @brief Returns this element as a scalar.
 *
 * @return A const reference to the stored scalar.
 */
const Scalar& Element::as_scalar() const {
    return std::get<Scalar>(value);
}

/**
 * @brief Reads the top-level AnnData object from a modern `.h5ad` file.
 *
 * The root group is validated first, then each well-known AnnData slot is read
 * if present. Optional mappings are only decoded when the corresponding links
 * exist in the file.
 *
 * @param path The filesystem path to the `.h5ad` file to parse.
 * @return The decoded top-level AnnData object.
 */
AnnData read_h5ad(const std::filesystem::path& path) {
    const Handle file = open_file_readonly(path);

    const std::string root_type = read_string_attribute(file.get(), "encoding-type", "/");
    const std::string root_version = read_string_attribute(file.get(), "encoding-version", "/");
    if (root_type != "anndata") {
        throw Error("root encoding-type is not anndata");
    }
    if (root_version != "0.1.0") {
        throw Error("unsupported anndata encoding-version: " + root_version);
    }

    AnnData adata;
    if (link_exists(file.get(), "/X")) {
        adata.X = read_element(file.get(), "/X");
    }
    adata.obs = read_dataframe(file.get(), "/obs");
    adata.var = read_dataframe(file.get(), "/var");
    if (link_exists(file.get(), "/obsm")) {
        adata.obsm = read_mapping(file.get(), "/obsm", "dict");
    }
    if (link_exists(file.get(), "/varm")) {
        adata.varm = read_mapping(file.get(), "/varm", "dict");
    }
    if (link_exists(file.get(), "/obsp")) {
        adata.obsp = read_mapping(file.get(), "/obsp", "dict");
    }
    if (link_exists(file.get(), "/varp")) {
        adata.varp = read_mapping(file.get(), "/varp", "dict");
    }
    if (link_exists(file.get(), "/layers")) {
        adata.layers = read_mapping(file.get(), "/layers", "dict");
    }
    if (link_exists(file.get(), "/uns")) {
        adata.uns = read_mapping(file.get(), "/uns", "dict");
    }
    if (link_exists(file.get(), "/raw")) {
        adata.raw = read_mapping(file.get(), "/raw", "raw");
    }
    return adata;
}

}  // namespace anndata_cpp

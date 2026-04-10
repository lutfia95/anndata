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

std::string join_path(std::string_view base, std::string_view leaf) {
    if (base.empty() || base == "/") {
        return "/" + std::string(leaf);
    }
    return std::string(base) + "/" + std::string(leaf);
}

void expect_ok(h5::herr_t status, const std::string& message) {
    if (status < 0) {
        throw Error(message);
    }
}

void expect_valid(hid_t id, const std::string& message) {
    if (id < 0) {
        throw Error(message);
    }
}

Handle open_file_readonly(const std::filesystem::path& path) {
    h5::initialize();
    const hid_t file_id = h5::H5Fopen(path.c_str(), h5::kFileAccRdOnly, h5::kDefault);
    expect_valid(file_id, "failed to open h5ad file: " + path.string());
    return Handle(file_id, &h5::H5Fclose);
}

Handle open_object(hid_t loc_id, const std::string& path) {
    const hid_t object_id = h5::H5Oopen(loc_id, path.c_str(), h5::kDefault);
    expect_valid(object_id, "failed to open HDF5 object at " + path);
    return Handle(object_id, &h5::H5Oclose);
}

Handle open_group(hid_t loc_id, const std::string& path) {
    const hid_t group_id = h5::H5Gopen2(loc_id, path.c_str(), h5::kDefault);
    expect_valid(group_id, "failed to open HDF5 group at " + path);
    return Handle(group_id, &h5::H5Gclose);
}

Handle open_dataset(hid_t loc_id, const std::string& path) {
    const hid_t dataset_id = h5::H5Dopen2(loc_id, path.c_str(), h5::kDefault);
    expect_valid(dataset_id, "failed to open HDF5 dataset at " + path);
    return Handle(dataset_id, &h5::H5Dclose);
}

Handle open_attribute(hid_t obj_id, const std::string& name, const std::string& context) {
    if (h5::H5Aexists(obj_id, name.c_str()) <= 0) {
        throw Error(context + " is missing required attribute '" + name + "'");
    }
    const hid_t attr_id = h5::H5Aopen_name(obj_id, name.c_str());
    expect_valid(attr_id, "failed to open attribute '" + name + "' on " + context);
    return Handle(attr_id, &h5::H5Aclose);
}

Handle get_dataset_type(hid_t dataset_id, const std::string& context) {
    const hid_t type_id = h5::H5Dget_type(dataset_id);
    expect_valid(type_id, "failed to read datatype for " + context);
    return Handle(type_id, &h5::H5Tclose);
}

Handle get_dataset_space(hid_t dataset_id, const std::string& context) {
    const hid_t space_id = h5::H5Dget_space(dataset_id);
    expect_valid(space_id, "failed to read dataspace for " + context);
    return Handle(space_id, &h5::H5Sclose);
}

Handle get_attribute_type(hid_t attr_id, const std::string& context) {
    const hid_t type_id = h5::H5Aget_type(attr_id);
    expect_valid(type_id, "failed to read attribute datatype for " + context);
    return Handle(type_id, &h5::H5Tclose);
}

Handle get_attribute_space(hid_t attr_id, const std::string& context) {
    const hid_t space_id = h5::H5Aget_space(attr_id);
    expect_valid(space_id, "failed to read attribute dataspace for " + context);
    return Handle(space_id, &h5::H5Sclose);
}

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

std::string trim_c_string(const char* bytes, std::size_t width) {
    std::size_t length = 0;
    while (length < width && bytes[length] != '\0') {
        ++length;
    }
    return std::string(bytes, length);
}

bool link_exists(hid_t loc_id, const std::string& path) {
    return h5::H5Lexists(loc_id, path.c_str(), h5::kDefault) > 0;
}

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

bool read_bool_attribute(hid_t obj_id, const std::string& name, const std::string& context) {
    const auto values = read_int_vector_attribute(obj_id, name, context);
    if (values.size() != 1) {
        throw Error("attribute '" + name + "' on " + context + " is not scalar");
    }
    return values[0] != 0;
}

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

std::size_t NumericArray::element_count() const {
    return product(shape);
}

std::size_t NumericArray::item_size() const {
    return anndata_cpp::item_size(dtype);
}

bool Scalar::is_bool() const {
    return std::holds_alternative<bool>(value);
}

bool Scalar::is_int() const {
    return std::holds_alternative<std::int64_t>(value);
}

bool Scalar::is_double() const {
    return std::holds_alternative<double>(value);
}

bool Scalar::is_string() const {
    return std::holds_alternative<std::string>(value);
}

bool Scalar::as_bool() const {
    return std::get<bool>(value);
}

std::int64_t Scalar::as_int() const {
    return std::get<std::int64_t>(value);
}

double Scalar::as_double() const {
    return std::get<double>(value);
}

const std::string& Scalar::as_string() const {
    return std::get<std::string>(value);
}

bool Element::is_numeric_array() const {
    return std::holds_alternative<NumericArray>(value);
}

bool Element::is_string_array() const {
    return std::holds_alternative<StringArray>(value);
}

bool Element::is_categorical() const {
    return std::holds_alternative<Categorical>(value);
}

bool Element::is_sparse() const {
    return std::holds_alternative<SparseMatrix>(value);
}

bool Element::is_dataframe() const {
    return std::holds_alternative<DataFrame>(value);
}

bool Element::is_mapping() const {
    return std::holds_alternative<std::shared_ptr<Mapping>>(value);
}

bool Element::is_scalar() const {
    return std::holds_alternative<Scalar>(value);
}

const NumericArray& Element::as_numeric_array() const {
    return std::get<NumericArray>(value);
}

const StringArray& Element::as_string_array() const {
    return std::get<StringArray>(value);
}

const Categorical& Element::as_categorical() const {
    return std::get<Categorical>(value);
}

const SparseMatrix& Element::as_sparse() const {
    return std::get<SparseMatrix>(value);
}

const DataFrame& Element::as_dataframe() const {
    return std::get<DataFrame>(value);
}

const Mapping& Element::as_mapping() const {
    return *std::get<std::shared_ptr<Mapping>>(value);
}

const Scalar& Element::as_scalar() const {
    return std::get<Scalar>(value);
}

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

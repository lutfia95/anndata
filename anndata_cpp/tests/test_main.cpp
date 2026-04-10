#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "anndata_cpp/anndata.hpp"
#include "hdf5_api.hpp"

namespace fs = std::filesystem;

namespace {

using anndata_cpp::AnnData;
using anndata_cpp::Categorical;
using anndata_cpp::Column;
using anndata_cpp::Element;
using anndata_cpp::Error;
using anndata_cpp::NumericArray;
using anndata_cpp::Scalar;
using anndata_cpp::SparseMatrix;
using anndata_cpp::StringArray;

/**
 * @brief Owns an HDF5 identifier used while constructing test fixtures.
 *
 * The test suite writes synthetic `.h5ad` files directly through the HDF5 C
 * API, so this RAII wrapper mirrors the production parser's handle management.
 */
class Handle {
public:
    using CloseFn = anndata_cpp::h5::herr_t (*)(anndata_cpp::h5::hid_t);

    Handle() = default;
    Handle(anndata_cpp::h5::hid_t id, CloseFn close_fn) : id_(id), close_fn_(close_fn) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    Handle(Handle&& other) noexcept : id_(other.id_), close_fn_(other.close_fn_) {
        other.id_ = -1;
        other.close_fn_ = nullptr;
    }

    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            if (id_ >= 0 && close_fn_ != nullptr) {
                close_fn_(id_);
            }
            id_ = other.id_;
            close_fn_ = other.close_fn_;
            other.id_ = -1;
            other.close_fn_ = nullptr;
        }
        return *this;
    }

    ~Handle() {
        if (id_ >= 0 && close_fn_ != nullptr) {
            close_fn_(id_);
        }
    }

    anndata_cpp::h5::hid_t get() const {
        return id_;
    }

private:
    anndata_cpp::h5::hid_t id_ = -1;
    CloseFn close_fn_ = nullptr;
};

/**
 * @brief Asserts that a boolean condition holds during a test.
 *
 * @param condition The condition that must evaluate to `true`.
 * @param message The failure message to raise when the condition is false.
 */
void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
/**
 * @brief Asserts that two values compare equal.
 *
 * @tparam T The comparable value type.
 * @param actual The computed value under test.
 * @param expected The expected reference value.
 * @param message The failure message to raise on mismatch.
 */
void expect_equal(const T& actual, const T& expected, const std::string& message) {
    if (!(actual == expected)) {
        throw std::runtime_error(message);
    }
}

/**
 * @brief Asserts that an HDF5 status code indicates success.
 *
 * @param status The status value returned by an HDF5 call.
 * @param message The failure message to raise on error.
 */
void expect_ok(anndata_cpp::h5::herr_t status, const std::string& message) {
    if (status < 0) {
        throw std::runtime_error(message);
    }
}

/**
 * @brief Asserts that an HDF5 identifier is valid.
 *
 * @param id The identifier returned by an HDF5 open or create call.
 * @param message The failure message to raise on error.
 */
void expect_valid(anndata_cpp::h5::hid_t id, const std::string& message) {
    if (id < 0) {
        throw std::runtime_error(message);
    }
}

/**
 * @brief Maps a local numeric dtype to the matching HDF5 native type.
 *
 * The fixture writers use this helper when creating numeric datasets of
 * different dtypes directly through the HDF5 C API.
 *
 * @param dtype The local numeric dtype to map.
 * @return The corresponding HDF5 native memory type identifier.
 */
anndata_cpp::h5::hid_t native_type_for(NumericArray::DType dtype) {
    using namespace anndata_cpp::h5;
    switch (dtype) {
        case NumericArray::DType::kBool:
            return H5T_NATIVE_UCHAR_g;
        case NumericArray::DType::kUInt8:
            return H5T_NATIVE_UINT8_g;
        case NumericArray::DType::kUInt16:
            return H5T_NATIVE_UINT16_g;
        case NumericArray::DType::kUInt32:
            return H5T_NATIVE_UINT32_g;
        case NumericArray::DType::kUInt64:
            return H5T_NATIVE_UINT64_g;
        case NumericArray::DType::kInt8:
            return H5T_NATIVE_INT8_g;
        case NumericArray::DType::kInt16:
            return H5T_NATIVE_INT16_g;
        case NumericArray::DType::kInt32:
            return H5T_NATIVE_INT32_g;
        case NumericArray::DType::kInt64:
            return H5T_NATIVE_INT64_g;
        case NumericArray::DType::kFloat32:
            return H5T_NATIVE_FLOAT_g;
        case NumericArray::DType::kFloat64:
            return H5T_NATIVE_DOUBLE_g;
    }
    throw std::runtime_error("unsupported dtype");
}

/**
 * @brief Creates a fresh HDF5 file for a synthetic test fixture.
 *
 * Parent directories are created automatically before the file is truncated.
 *
 * @param path The filesystem path where the fixture file should be written.
 * @return An owning handle for the created file.
 */
Handle make_file(const fs::path& path) {
    anndata_cpp::h5::initialize();
    fs::create_directories(path.parent_path());
    const auto id = anndata_cpp::h5::H5Fcreate(
        path.c_str(),
        anndata_cpp::h5::kFileAccTrunc,
        anndata_cpp::h5::kDefault,
        anndata_cpp::h5::kDefault
    );
    expect_valid(id, "failed to create HDF5 test file");
    return Handle(id, &anndata_cpp::h5::H5Fclose);
}

/**
 * @brief Creates a group inside a synthetic fixture file.
 *
 * @param loc_id The parent HDF5 location.
 * @param path The absolute path of the group to create.
 * @return An owning handle for the created group.
 */
Handle make_group(anndata_cpp::h5::hid_t loc_id, const std::string& path) {
    const auto id = anndata_cpp::h5::H5Gcreate2(
        loc_id,
        path.c_str(),
        anndata_cpp::h5::kDefault,
        anndata_cpp::h5::kDefault,
        anndata_cpp::h5::kDefault
    );
    expect_valid(id, "failed to create group " + path);
    return Handle(id, &anndata_cpp::h5::H5Gclose);
}

/**
 * @brief Creates a dataspace for a synthetic dataset or attribute.
 *
 * Empty shapes are represented as scalar dataspaces so the fixture writers can
 * create scalar datasets and attributes in the same way as arrays.
 *
 * @param shape The desired logical shape.
 * @return An owning handle for the created dataspace.
 */
Handle make_dataspace(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        const auto id = anndata_cpp::h5::H5Screate(anndata_cpp::h5::kScalarSpace);
        expect_valid(id, "failed to create scalar dataspace");
        return Handle(id, &anndata_cpp::h5::H5Sclose);
    }
    std::vector<anndata_cpp::h5::hsize_t> dims;
    dims.reserve(shape.size());
    for (const auto value : shape) {
        dims.push_back(static_cast<anndata_cpp::h5::hsize_t>(value));
    }
    const auto id = anndata_cpp::h5::H5Screate_simple(
        static_cast<int>(dims.size()),
        dims.data(),
        nullptr
    );
    expect_valid(id, "failed to create dataspace");
    return Handle(id, &anndata_cpp::h5::H5Sclose);
}

/**
 * @brief Creates a variable-length UTF-8 HDF5 string datatype.
 *
 * @return An owning handle for the configured string datatype.
 */
Handle make_vlen_utf8_string_type() {
    auto type = Handle(anndata_cpp::h5::H5Tcopy(anndata_cpp::h5::H5T_C_S1_g), &anndata_cpp::h5::H5Tclose);
    expect_valid(type.get(), "failed to copy string type");
    expect_ok(anndata_cpp::h5::H5Tset_size(type.get(), anndata_cpp::h5::kVariable), "failed to set variable string size");
    expect_ok(anndata_cpp::h5::H5Tset_cset(type.get(), anndata_cpp::h5::kCsetUtf8), "failed to set string cset");
    expect_ok(anndata_cpp::h5::H5Tset_strpad(type.get(), anndata_cpp::h5::kStrNullterm), "failed to set string padding");
    return type;
}

/**
 * @brief Writes a scalar string attribute onto an HDF5 object.
 *
 * @param obj_id The target object identifier.
 * @param name The attribute name to create.
 * @param value The string value to write.
 */
void write_string_attribute(anndata_cpp::h5::hid_t obj_id, const std::string& name, const std::string& value) {
    auto type = make_vlen_utf8_string_type();
    auto space = make_dataspace({});
    auto attr = Handle(
        anndata_cpp::h5::H5Acreate2(obj_id, name.c_str(), type.get(), space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Aclose
    );
    expect_valid(attr.get(), "failed to create string attribute " + name);
    const char* raw = value.c_str();
    expect_ok(anndata_cpp::h5::H5Awrite(attr.get(), type.get(), &raw), "failed to write string attribute " + name);
}

/**
 * @brief Writes a string-vector attribute onto an HDF5 object.
 *
 * This helper is used for metadata such as dataframe column order.
 *
 * @param obj_id The target object identifier.
 * @param name The attribute name to create.
 * @param values The string values to store.
 */
void write_string_vector_attribute(
    anndata_cpp::h5::hid_t obj_id,
    const std::string& name,
    const std::vector<std::string>& values
) {
    auto type = make_vlen_utf8_string_type();
    auto space = make_dataspace({values.size()});
    auto attr = Handle(
        anndata_cpp::h5::H5Acreate2(obj_id, name.c_str(), type.get(), space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Aclose
    );
    expect_valid(attr.get(), "failed to create string vector attribute " + name);
    std::vector<const char*> raw;
    raw.reserve(values.size());
    for (const auto& value : values) {
        raw.push_back(value.c_str());
    }
    if (!raw.empty()) {
        expect_ok(anndata_cpp::h5::H5Awrite(attr.get(), type.get(), raw.data()), "failed to write string vector attribute " + name);
    }
}

/**
 * @brief Writes an integer vector attribute onto an HDF5 object.
 *
 * @param obj_id The target object identifier.
 * @param name The attribute name to create.
 * @param values The integer values to store.
 */
void write_int_vector_attribute(
    anndata_cpp::h5::hid_t obj_id,
    const std::string& name,
    const std::vector<std::int64_t>& values
) {
    auto space = make_dataspace({values.size()});
    auto attr = Handle(
        anndata_cpp::h5::H5Acreate2(obj_id, name.c_str(), anndata_cpp::h5::H5T_NATIVE_INT64_g, space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Aclose
    );
    expect_valid(attr.get(), "failed to create integer vector attribute " + name);
    if (!values.empty()) {
        expect_ok(anndata_cpp::h5::H5Awrite(attr.get(), anndata_cpp::h5::H5T_NATIVE_INT64_g, values.data()), "failed to write integer vector attribute " + name);
    }
}

/**
 * @brief Writes a boolean attribute onto an HDF5 object.
 *
 * @param obj_id The target object identifier.
 * @param name The attribute name to create.
 * @param value The boolean value to store.
 */
void write_bool_attribute(anndata_cpp::h5::hid_t obj_id, const std::string& name, bool value) {
    auto space = make_dataspace({});
    auto attr = Handle(
        anndata_cpp::h5::H5Acreate2(obj_id, name.c_str(), anndata_cpp::h5::H5T_NATIVE_UCHAR_g, space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Aclose
    );
    expect_valid(attr.get(), "failed to create bool attribute " + name);
    const std::uint8_t raw = value ? 1U : 0U;
    expect_ok(anndata_cpp::h5::H5Awrite(attr.get(), anndata_cpp::h5::H5T_NATIVE_UCHAR_g, &raw), "failed to write bool attribute " + name);
}

template <typename T>
/**
 * @brief Writes a raw numeric dataset without AnnData array annotations.
 *
 * This helper is primarily used when constructing sparse payload arrays such as
 * `data`, `indices`, and `indptr`.
 *
 * @tparam T The concrete numeric value type being written.
 * @param loc_id The parent HDF5 location.
 * @param path The dataset path to create.
 * @param dtype The logical dtype to encode.
 * @param shape The dataset shape.
 * @param values The flat numeric payload to write.
 */
void write_numeric_dataset(
    anndata_cpp::h5::hid_t loc_id,
    const std::string& path,
    NumericArray::DType dtype,
    const std::vector<std::size_t>& shape,
    const std::vector<T>& values
) {
    auto space = make_dataspace(shape);
    auto dataset = Handle(
        anndata_cpp::h5::H5Dcreate2(loc_id, path.c_str(), native_type_for(dtype), space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Dclose
    );
    expect_valid(dataset.get(), "failed to create numeric dataset " + path);
    if (!values.empty()) {
        expect_ok(
            anndata_cpp::h5::H5Dwrite(dataset.get(), native_type_for(dtype), anndata_cpp::h5::kAll, anndata_cpp::h5::kAll, anndata_cpp::h5::kDefault, values.data()),
            "failed to write numeric dataset " + path
        );
    }
}

/**
 * @brief Attaches modern dense-array encoding metadata to a dataset.
 *
 * @param object_id The dataset identifier to annotate.
 */
void annotate_array(anndata_cpp::h5::hid_t object_id) {
    write_string_attribute(object_id, "encoding-type", "array");
    write_string_attribute(object_id, "encoding-version", "0.2.0");
}

template <typename T>
/**
 * @brief Writes an AnnData tagged dense numeric array dataset.
 *
 * @tparam T The concrete numeric value type being written.
 * @param loc_id The parent HDF5 location.
 * @param path The dataset path to create.
 * @param dtype The logical numeric dtype to encode.
 * @param shape The dataset shape.
 * @param values The flat numeric payload to write.
 */
void write_array_dataset(
    anndata_cpp::h5::hid_t loc_id,
    const std::string& path,
    NumericArray::DType dtype,
    const std::vector<std::size_t>& shape,
    const std::vector<T>& values
) {
    auto space = make_dataspace(shape);
    auto dataset = Handle(
        anndata_cpp::h5::H5Dcreate2(loc_id, path.c_str(), native_type_for(dtype), space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Dclose
    );
    expect_valid(dataset.get(), "failed to create array dataset " + path);
    annotate_array(dataset.get());
    if (!values.empty()) {
        expect_ok(
            anndata_cpp::h5::H5Dwrite(dataset.get(), native_type_for(dtype), anndata_cpp::h5::kAll, anndata_cpp::h5::kAll, anndata_cpp::h5::kDefault, values.data()),
            "failed to write array dataset " + path
        );
    }
}

/**
 * @brief Writes an AnnData tagged dense string array dataset.
 *
 * @param loc_id The parent HDF5 location.
 * @param path The dataset path to create.
 * @param shape The dataset shape.
 * @param values The flat string payload to write.
 */
void write_string_array_dataset(
    anndata_cpp::h5::hid_t loc_id,
    const std::string& path,
    const std::vector<std::size_t>& shape,
    const std::vector<std::string>& values
) {
    auto type = make_vlen_utf8_string_type();
    auto space = make_dataspace(shape);
    auto dataset = Handle(
        anndata_cpp::h5::H5Dcreate2(loc_id, path.c_str(), type.get(), space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Dclose
    );
    expect_valid(dataset.get(), "failed to create string-array dataset " + path);
    write_string_attribute(dataset.get(), "encoding-type", "string-array");
    write_string_attribute(dataset.get(), "encoding-version", "0.2.0");
    std::vector<const char*> raw;
    raw.reserve(values.size());
    for (const auto& value : values) {
        raw.push_back(value.c_str());
    }
    if (!raw.empty()) {
        expect_ok(
            anndata_cpp::h5::H5Dwrite(dataset.get(), type.get(), anndata_cpp::h5::kAll, anndata_cpp::h5::kAll, anndata_cpp::h5::kDefault, raw.data()),
            "failed to write string-array dataset " + path
        );
    }
}

template <typename T>
/**
 * @brief Writes a tagged numeric scalar dataset.
 *
 * @tparam T The concrete scalar value type being written.
 * @param loc_id The parent HDF5 location.
 * @param path The dataset path to create.
 * @param dtype The logical numeric dtype to encode.
 * @param value The scalar value to write.
 */
void write_numeric_scalar_dataset(
    anndata_cpp::h5::hid_t loc_id,
    const std::string& path,
    NumericArray::DType dtype,
    const T& value
) {
    auto space = make_dataspace({});
    auto dataset = Handle(
        anndata_cpp::h5::H5Dcreate2(loc_id, path.c_str(), native_type_for(dtype), space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Dclose
    );
    expect_valid(dataset.get(), "failed to create numeric scalar dataset " + path);
    write_string_attribute(dataset.get(), "encoding-type", "numeric-scalar");
    write_string_attribute(dataset.get(), "encoding-version", "0.2.0");
    expect_ok(
        anndata_cpp::h5::H5Dwrite(dataset.get(), native_type_for(dtype), anndata_cpp::h5::kAll, anndata_cpp::h5::kAll, anndata_cpp::h5::kDefault, &value),
        "failed to write numeric scalar dataset " + path
    );
}

/**
 * @brief Writes a tagged string scalar dataset.
 *
 * @param loc_id The parent HDF5 location.
 * @param path The dataset path to create.
 * @param value The string scalar value to write.
 */
void write_string_scalar_dataset(
    anndata_cpp::h5::hid_t loc_id,
    const std::string& path,
    const std::string& value
) {
    auto type = make_vlen_utf8_string_type();
    auto space = make_dataspace({});
    auto dataset = Handle(
        anndata_cpp::h5::H5Dcreate2(loc_id, path.c_str(), type.get(), space.get(), anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault, anndata_cpp::h5::kDefault),
        &anndata_cpp::h5::H5Dclose
    );
    expect_valid(dataset.get(), "failed to create string scalar dataset " + path);
    write_string_attribute(dataset.get(), "encoding-type", "string");
    write_string_attribute(dataset.get(), "encoding-version", "0.2.0");
    const char* raw = value.c_str();
    expect_ok(
        anndata_cpp::h5::H5Dwrite(dataset.get(), type.get(), anndata_cpp::h5::kAll, anndata_cpp::h5::kAll, anndata_cpp::h5::kDefault, &raw),
        "failed to write string scalar dataset " + path
    );
}

/**
 * @brief Creates and annotates a dataframe group.
 *
 * The caller is responsible for writing the index dataset and any column
 * datasets beneath the group after the metadata is established.
 *
 * @param loc_id The parent HDF5 location.
 * @param path The dataframe group path to create.
 * @param index_name The stored dataframe index key.
 * @param column_order The ordered dataframe column names.
 */
void write_dataframe_group(
    anndata_cpp::h5::hid_t loc_id,
    const std::string& path,
    const std::string& index_name,
    const std::vector<std::string>& column_order
) {
    auto group = make_group(loc_id, path);
    write_string_attribute(group.get(), "_index", index_name);
    write_string_vector_attribute(group.get(), "column-order", column_order);
    write_string_attribute(group.get(), "encoding-type", "dataframe");
    write_string_attribute(group.get(), "encoding-version", "0.2.0");
}

/**
 * @brief Creates and annotates a dictionary-style mapping group.
 *
 * @param loc_id The parent HDF5 location.
 * @param path The mapping group path to create.
 */
void write_dict_group(anndata_cpp::h5::hid_t loc_id, const std::string& path) {
    auto group = make_group(loc_id, path);
    write_string_attribute(group.get(), "encoding-type", "dict");
    write_string_attribute(group.get(), "encoding-version", "0.1.0");
}

/**
 * @brief Creates and annotates a sparse matrix group.
 *
 * The caller is responsible for writing the `data`, `indices`, and `indptr`
 * children after the sparse group metadata is created.
 *
 * @param loc_id The parent HDF5 location.
 * @param path The sparse group path to create.
 * @param format The sparse layout to encode.
 * @param shape The two-dimensional matrix shape.
 */
void write_sparse_group(
    anndata_cpp::h5::hid_t loc_id,
    const std::string& path,
    SparseMatrix::Format format,
    std::pair<std::size_t, std::size_t> shape
) {
    auto group = make_group(loc_id, path);
    write_string_attribute(group.get(), "encoding-type", format == SparseMatrix::Format::kCsr ? "csr_matrix" : "csc_matrix");
    write_string_attribute(group.get(), "encoding-version", "0.1.0");
    write_int_vector_attribute(
        group.get(),
        "shape",
        {
            static_cast<std::int64_t>(shape.first),
            static_cast<std::int64_t>(shape.second),
        }
    );
}

/**
 * @brief Creates and annotates a categorical column group.
 *
 * @param loc_id The parent HDF5 location.
 * @param path The categorical group path to create.
 * @param codes The categorical integer codes.
 * @param categories The category labels.
 * @param ordered Whether the categorical values are ordered.
 */
void write_categorical_group(
    anndata_cpp::h5::hid_t loc_id,
    const std::string& path,
    const std::vector<std::int32_t>& codes,
    const std::vector<std::string>& categories,
    bool ordered
) {
    auto group = make_group(loc_id, path);
    write_string_attribute(group.get(), "encoding-type", "categorical");
    write_string_attribute(group.get(), "encoding-version", "0.2.0");
    write_bool_attribute(group.get(), "ordered", ordered);
    write_array_dataset(group.get(), "codes", NumericArray::DType::kInt32, {codes.size()}, codes);
    write_string_array_dataset(group.get(), "categories", {categories.size()}, categories);
}

/**
 * @brief Builds a synthetic dense modern `.h5ad` fixture.
 *
 * The fixture exercises dense `X`, numeric and categorical `obs` columns,
 * string `var` indexes, dense `obsm`, dense `layers`, and scalar `uns` values.
 *
 * @param root The directory where the fixture file should be written.
 * @return The path to the generated dense fixture.
 */
fs::path make_modern_dense_fixture(const fs::path& root) {
    const fs::path path = root / "modern_dense.h5ad";
    auto file = make_file(path);

    write_string_attribute(file.get(), "encoding-type", "anndata");
    write_string_attribute(file.get(), "encoding-version", "0.1.0");

    write_array_dataset(file.get(), "/X", NumericArray::DType::kFloat32, {2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});

    write_dataframe_group(file.get(), "/obs", "_index", {"score", "cluster"});
    write_string_array_dataset(file.get(), "/obs/_index", {2}, {"cell0", "cell1"});
    write_array_dataset(file.get(), "/obs/score", NumericArray::DType::kFloat64, {2}, std::vector<double>{0.5, 1.5});
    write_categorical_group(file.get(), "/obs/cluster", {0, 1}, {"A", "B"}, false);

    write_dataframe_group(file.get(), "/var", "_index", {});
    write_string_array_dataset(file.get(), "/var/_index", {3}, {"gene0", "gene1", "gene2"});

    write_dict_group(file.get(), "/obsm");
    write_array_dataset(file.get(), "/obsm/embed", NumericArray::DType::kFloat64, {2, 2}, std::vector<double>{10.0, 11.0, 12.0, 13.0});

    write_dict_group(file.get(), "/layers");
    write_array_dataset(file.get(), "/layers/norm", NumericArray::DType::kFloat32, {2, 3}, std::vector<float>{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F});

    write_dict_group(file.get(), "/uns");
    write_numeric_scalar_dataset(file.get(), "/uns/n_neighbors", NumericArray::DType::kInt64, static_cast<std::int64_t>(15));
    write_string_scalar_dataset(file.get(), "/uns/metric", "euclidean");

    return path;
}

/**
 * @brief Builds a synthetic sparse modern `.h5ad` fixture.
 *
 * The fixture exercises sparse `X`, sparse layers, and empty-but-present
 * mapping slots used by the parser.
 *
 * @param root The directory where the fixture file should be written.
 * @return The path to the generated sparse fixture.
 */
fs::path make_modern_sparse_fixture(const fs::path& root) {
    const fs::path path = root / "modern_sparse.h5ad";
    auto file = make_file(path);

    write_string_attribute(file.get(), "encoding-type", "anndata");
    write_string_attribute(file.get(), "encoding-version", "0.1.0");

    write_sparse_group(file.get(), "/X", SparseMatrix::Format::kCsr, {2, 3});
    write_array_dataset(file.get(), "/X/data", NumericArray::DType::kFloat32, {3}, std::vector<float>{1.0F, 2.0F, 3.0F});
    write_array_dataset(file.get(), "/X/indices", NumericArray::DType::kInt64, {3}, std::vector<std::int64_t>{0, 2, 1});
    write_array_dataset(file.get(), "/X/indptr", NumericArray::DType::kInt64, {3}, std::vector<std::int64_t>{0, 2, 3});

    write_dataframe_group(file.get(), "/obs", "_index", {});
    write_string_array_dataset(file.get(), "/obs/_index", {2}, {"cell0", "cell1"});

    write_dataframe_group(file.get(), "/var", "_index", {});
    write_string_array_dataset(file.get(), "/var/_index", {3}, {"gene0", "gene1", "gene2"});

    write_dict_group(file.get(), "/layers");
    write_sparse_group(file.get(), "/layers/counts", SparseMatrix::Format::kCsr, {2, 3});
    write_array_dataset(file.get(), "/layers/counts/data", NumericArray::DType::kFloat32, {1}, std::vector<float>{5.0F});
    write_array_dataset(file.get(), "/layers/counts/indices", NumericArray::DType::kInt64, {1}, std::vector<std::int64_t>{2});
    write_array_dataset(file.get(), "/layers/counts/indptr", NumericArray::DType::kInt64, {3}, std::vector<std::int64_t>{0, 0, 1});

    write_dict_group(file.get(), "/obsm");
    write_dict_group(file.get(), "/obsp");
    write_dict_group(file.get(), "/varm");
    write_dict_group(file.get(), "/varp");
    write_dict_group(file.get(), "/uns");

    return path;
}

/**
 * @brief Extracts a numeric dataframe column from a `Column` variant.
 *
 * @param column The column variant to unwrap.
 * @return A const reference to the stored numeric array.
 */
const NumericArray& as_numeric(const Column& column) {
    return std::get<NumericArray>(column);
}

/**
 * @brief Extracts a string dataframe column from a `Column` variant.
 *
 * @param column The column variant to unwrap.
 * @return A const reference to the stored string array.
 */
const StringArray& as_string_array(const Column& column) {
    return std::get<StringArray>(column);
}

/**
 * @brief Extracts a categorical dataframe column from a `Column` variant.
 *
 * @param column The column variant to unwrap.
 * @return A const reference to the stored categorical column.
 */
const Categorical& as_categorical(const Column& column) {
    return std::get<Categorical>(column);
}

/**
 * @brief Verifies parsing of a synthetic dense modern `.h5ad` file.
 *
 * This test covers dense `X`, dataframe columns, categoricals, dense mappings,
 * and scalar `uns` payloads.
 */
void test_synthetic_modern_dense_h5ad() {
    const fs::path temp_dir = fs::temp_directory_path() / "anndata_cpp_h5ad_dense";
    fs::remove_all(temp_dir);
    const fs::path fixture = make_modern_dense_fixture(temp_dir);

    const AnnData adata = anndata_cpp::read_h5ad(fixture);

    expect(adata.X.has_value(), "dense fixture should have X");
    expect(adata.X->is_numeric_array(), "X should be a dense numeric array");
    expect_equal(adata.X->as_numeric_array().shape, std::vector<std::size_t>({2, 3}), "unexpected dense X shape");
    expect_equal(adata.X->as_numeric_array().values<float>(), std::vector<float>({1, 2, 3, 4, 5, 6}), "unexpected dense X values");

    expect_equal(as_string_array(adata.obs.index).values, std::vector<std::string>({"cell0", "cell1"}), "unexpected obs index");
    expect_equal(as_numeric(adata.obs.columns.at("score")).values<double>(), std::vector<double>({0.5, 1.5}), "unexpected obs score");
    expect_equal(as_categorical(adata.obs.columns.at("cluster")).categories.values, std::vector<std::string>({"A", "B"}), "unexpected cluster categories");
    expect_equal(as_categorical(adata.obs.columns.at("cluster")).codes.values<std::int32_t>(), std::vector<std::int32_t>({0, 1}), "unexpected cluster codes");

    expect_equal(as_string_array(adata.var.index).values, std::vector<std::string>({"gene0", "gene1", "gene2"}), "unexpected var index");

    expect(adata.obsm != nullptr, "dense fixture should have obsm");
    expect(adata.obsm->items.count("embed") == 1, "obsm/embed missing");
    expect_equal(adata.obsm->items.at("embed").as_numeric_array().values<double>(), std::vector<double>({10.0, 11.0, 12.0, 13.0}), "unexpected obsm/embed values");

    expect(adata.layers != nullptr, "dense fixture should have layers");
    expect(adata.layers->items.count("norm") == 1, "layers/norm missing");
    expect_equal(adata.layers->items.at("norm").as_numeric_array().values<float>(), std::vector<float>({0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F}), "unexpected layers/norm values");

    expect(adata.uns != nullptr, "dense fixture should have uns");
    expect(adata.uns->items.at("n_neighbors").as_scalar().as_int() == 15, "unexpected uns/n_neighbors");
    expect_equal(adata.uns->items.at("metric").as_scalar().as_string(), std::string("euclidean"), "unexpected uns/metric");
}

/**
 * @brief Verifies parsing of a synthetic sparse modern `.h5ad` file.
 *
 * This test covers CSR `X`, sparse layers, and the presence of the standard
 * mapping slots used in modern AnnData layouts.
 */
void test_synthetic_modern_sparse_h5ad() {
    const fs::path temp_dir = fs::temp_directory_path() / "anndata_cpp_h5ad_sparse";
    fs::remove_all(temp_dir);
    const fs::path fixture = make_modern_sparse_fixture(temp_dir);

    const AnnData adata = anndata_cpp::read_h5ad(fixture);

    expect(adata.X.has_value(), "sparse fixture should have X");
    expect(adata.X->is_sparse(), "X should be sparse");
    const auto& x = adata.X->as_sparse();
    expect(x.format == SparseMatrix::Format::kCsr, "X should be CSR");
    expect(x.shape.first == 2 && x.shape.second == 3, "unexpected sparse X shape");
    expect_equal(x.data.values<float>(), std::vector<float>({1.0F, 2.0F, 3.0F}), "unexpected sparse data");
    expect_equal(x.indices.values<std::int64_t>(), std::vector<std::int64_t>({0, 2, 1}), "unexpected sparse indices");
    expect_equal(x.indptr.values<std::int64_t>(), std::vector<std::int64_t>({0, 2, 3}), "unexpected sparse indptr");

    expect(adata.layers != nullptr, "sparse fixture should have layers");
    expect(adata.layers->items.count("counts") == 1, "layers/counts missing");
    const auto& counts = adata.layers->items.at("counts").as_sparse();
    expect_equal(counts.data.values<float>(), std::vector<float>({5.0F}), "unexpected sparse layer data");
    expect_equal(counts.indices.values<std::int64_t>(), std::vector<std::int64_t>({2}), "unexpected sparse layer indices");
    expect_equal(counts.indptr.values<std::int64_t>(), std::vector<std::int64_t>({0, 0, 1}), "unexpected sparse layer indptr");
}

/**
 * @brief Verifies parsing of the bundled real-world modern archive fixture.
 *
 * The current `v0.11.4` archive is intentionally simple but still acts as a
 * regression check against a file produced by the Python project itself.
 */
void test_real_archive_v0114_h5ad() {
    const fs::path archive =
        fs::path(ANNDATA_CPP_REPO_ROOT) / "tests" / "data" / "archives" / "v0.11.4" / "adata.h5ad";

    const AnnData adata = anndata_cpp::read_h5ad(archive);

    expect(!adata.X.has_value(), "v0.11.4 archive should not have X");
    expect_equal(as_string_array(adata.obs.index).values.front(), std::string("0"), "unexpected first obs index");
    expect_equal(as_string_array(adata.obs.index).values.back(), std::string("9"), "unexpected last obs index");
    expect_equal(as_string_array(adata.var.index).values.front(), std::string("0"), "unexpected first var index");
    expect_equal(as_string_array(adata.var.index).values.back(), std::string("19"), "unexpected last var index");
}

/**
 * @brief Verifies that older unsupported archive layouts fail cleanly.
 *
 * The project currently targets the modern tagged format only, so older files
 * should raise a clear error instead of being misinterpreted silently.
 */
void test_old_archive_fails_cleanly() {
    const fs::path archive =
        fs::path(ANNDATA_CPP_REPO_ROOT) / "tests" / "data" / "archives" / "v0.7.8" / "adata.h5ad";

    bool saw_error = false;
    try {
        static_cast<void>(anndata_cpp::read_h5ad(archive));
    } catch (const Error& error) {
        saw_error = true;
        const std::string message = error.what();
        expect(
            message.find("unsupported") != std::string::npos ||
                message.find("missing") != std::string::npos,
            "old archive failure should describe the unsupported format boundary"
        );
    }
    expect(saw_error, "old archive should currently fail");
}

}  // namespace

/**
 * @brief Runs the standalone C++ test suite without an external framework.
 *
 * Each test is executed sequentially, printing a pass/fail line and returning
 * a non-zero exit code on the first failure.
 *
 * @return `0` when every test passes, otherwise `1`.
 */
int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"synthetic_modern_dense_h5ad", test_synthetic_modern_dense_h5ad},
        {"synthetic_modern_sparse_h5ad", test_synthetic_modern_sparse_h5ad},
        {"real_archive_v0114_h5ad", test_real_archive_v0114_h5ad},
        {"old_archive_fails_cleanly", test_old_archive_fails_cleanly},
    };

    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }

    return 0;
}

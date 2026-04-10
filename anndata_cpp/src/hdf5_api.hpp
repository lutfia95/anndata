#pragma once

#include <cstddef>
#include <cstdint>

namespace anndata_cpp::h5 {

/**
 * @brief Stores the opaque identifier type used by the HDF5 C API.
 *
 * HDF5 represents files, groups, datasets, datatypes, and other handles with
 * integer identifiers. This alias keeps the local shim independent from the
 * system headers while matching the runtime ABI used on this platform.
 */
using hid_t = long long;

/**
 * @brief Stores the standard HDF5 status return type.
 *
 * Most HDF5 functions return a negative value on failure and a non-negative
 * value on success, so parser helpers use this alias for validation checks.
 */
using herr_t = int;
using htri_t = int;
using hsize_t = unsigned long long;
using hssize_t = long long;

constexpr hid_t kDefault = 0;
constexpr hid_t kAll = 0;
constexpr unsigned kFileAccRdOnly = 0x0000u;
constexpr unsigned kFileAccTrunc = 0x0002u;
constexpr std::size_t kVariable = static_cast<std::size_t>(-1);
constexpr int kCsetUtf8 = 1;
constexpr int kStrNullterm = 0;
constexpr int kScalarSpace = 0;

/**
 * @brief Enumerates the object kinds returned by `H5Iget_type`.
 *
 * The parser uses these values to distinguish files, groups, datasets,
 * datatypes, dataspaces, and attributes when dispatching element readers.
 */
enum class IType : int {
    kUninit = -2,
    kBadId = -1,
    kFile = 1,
    kGroup = 2,
    kDatatype = 3,
    kDataspace = 4,
    kDataset = 5,
    kAttr = 6,
};

/**
 * @brief Enumerates the datatype classes returned by `H5Tget_class`.
 *
 * These values allow the parser to recognize whether a dataset or attribute is
 * numeric, string-like, enum-based, or otherwise unsupported.
 */
enum class TClass : int {
    kNoClass = -1,
    kInteger = 0,
    kFloat = 1,
    kTime = 2,
    kString = 3,
    kBitfield = 4,
    kOpaque = 5,
    kCompound = 6,
    kReference = 7,
    kEnum = 8,
    kVlen = 9,
    kArray = 10,
};

/**
 * @brief Describes the signedness of an integer HDF5 datatype.
 *
 * Signedness is needed when mapping HDF5 integer types onto the local
 * `NumericArray::DType` enumeration.
 */
enum class TSign : int {
    kError = -1,
    kNone = 0,
    kTwosComplement = 1,
};

extern "C" {

/**
 * @brief Declares the subset of HDF5 C symbols used by the local parser.
 *
 * The project intentionally avoids depending on development headers in this
 * environment, so these declarations mirror the runtime functions and global
 * type identifiers required by the reader and test fixture writers.
 */

herr_t H5open(void);

hid_t H5Fopen(const char* filename, unsigned flags, hid_t access_plist);
hid_t H5Fcreate(const char* filename, unsigned flags, hid_t create_plist, hid_t access_plist);
herr_t H5Fclose(hid_t file_id);

hid_t H5Oopen(hid_t loc_id, const char* name, hid_t lapl_id);
herr_t H5Oclose(hid_t object_id);
IType H5Iget_type(hid_t object_id);

htri_t H5Lexists(hid_t loc_id, const char* name, hid_t lapl_id);

hid_t H5Gopen2(hid_t loc_id, const char* name, hid_t gapl_id);
hid_t H5Gcreate2(hid_t loc_id, const char* name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id);
herr_t H5Gclose(hid_t group_id);
herr_t H5Gget_num_objs(hid_t loc_id, hsize_t* num_objs);
long H5Gget_objname_by_idx(hid_t loc_id, hsize_t idx, char* name, std::size_t size);

hid_t H5Dopen2(hid_t loc_id, const char* name, hid_t dapl_id);
hid_t H5Dcreate2(
    hid_t loc_id,
    const char* name,
    hid_t type_id,
    hid_t space_id,
    hid_t lcpl_id,
    hid_t dcpl_id,
    hid_t dapl_id
);
herr_t H5Dclose(hid_t dataset_id);
hid_t H5Dget_type(hid_t dataset_id);
hid_t H5Dget_space(hid_t dataset_id);
herr_t H5Dread(
    hid_t dataset_id,
    hid_t mem_type_id,
    hid_t mem_space_id,
    hid_t file_space_id,
    hid_t xfer_plist_id,
    void* buf
);
herr_t H5Dwrite(
    hid_t dataset_id,
    hid_t mem_type_id,
    hid_t mem_space_id,
    hid_t file_space_id,
    hid_t xfer_plist_id,
    const void* buf
);
herr_t H5Dvlen_reclaim(hid_t type_id, hid_t space_id, hid_t plist_id, void* buf);

hid_t H5Aopen_name(hid_t loc_id, const char* attr_name);
htri_t H5Aexists(hid_t obj_id, const char* attr_name);
hid_t H5Acreate2(hid_t obj_id, const char* attr_name, hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id);
hid_t H5Aget_type(hid_t attr_id);
hid_t H5Aget_space(hid_t attr_id);
herr_t H5Aread(hid_t attr_id, hid_t mem_type_id, void* buf);
herr_t H5Awrite(hid_t attr_id, hid_t mem_type_id, const void* buf);
herr_t H5Aclose(hid_t attr_id);

hid_t H5Screate(int type);
hid_t H5Screate_simple(int rank, const hsize_t* dims, const hsize_t* maxdims);
herr_t H5Sclose(hid_t space_id);
int H5Sget_simple_extent_ndims(hid_t space_id);
int H5Sget_simple_extent_dims(hid_t space_id, hsize_t* dims, hsize_t* maxdims);

hid_t H5Tcopy(hid_t type_id);
herr_t H5Tclose(hid_t type_id);
TClass H5Tget_class(hid_t type_id);
std::size_t H5Tget_size(hid_t type_id);
TSign H5Tget_sign(hid_t type_id);
htri_t H5Tis_variable_str(hid_t type_id);
herr_t H5Tset_size(hid_t type_id, std::size_t size);
herr_t H5Tset_cset(hid_t type_id, int cset);
herr_t H5Tset_strpad(hid_t type_id, int strpad);

extern hid_t H5T_C_S1_g;
extern hid_t H5T_NATIVE_UCHAR_g;
extern hid_t H5T_NATIVE_UINT8_g;
extern hid_t H5T_NATIVE_UINT16_g;
extern hid_t H5T_NATIVE_UINT32_g;
extern hid_t H5T_NATIVE_UINT64_g;
extern hid_t H5T_NATIVE_INT8_g;
extern hid_t H5T_NATIVE_INT16_g;
extern hid_t H5T_NATIVE_INT32_g;
extern hid_t H5T_NATIVE_INT64_g;
extern hid_t H5T_NATIVE_FLOAT_g;
extern hid_t H5T_NATIVE_DOUBLE_g;

}  // extern "C"

/**
 * @brief Initializes the HDF5 runtime exactly once for the current process.
 *
 * The parser calls this helper before opening files so HDF5 global state is
 * ready. Repeated calls are safe because the initialization work is guarded by
 * a function-local static.
 */
inline void initialize() {
    static bool initialized = [] {
        H5open();
        return true;
    }();
    (void)initialized;
}

}  // namespace anndata_cpp::h5

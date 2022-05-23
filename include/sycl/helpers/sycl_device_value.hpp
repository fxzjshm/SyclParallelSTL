#ifndef __EXPERIMENTAL_DETAIL_SYCL_DEVICE_VALUE__
#define __EXPERIMENTAL_DETAIL_SYCL_DEVICE_VALUE__

#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_default_variables.hpp>
#include <sycl/helpers/sycl_device_pointer.hpp>

namespace sycl {
namespace helpers {

// forward declaration as the include order is unknown
template <typename T>
class device_pointer;

// reference: boost/compute/detail/buffer_value.hpp
template <typename T>
class device_value {
 public:
  device_value(T* ptr_) : ptr(ptr_) {}

#if defined(__SYCL_DEVICE_ONLY__) || defined(__SYCL_SINGLE_SOURCE__) || defined(__SYCL_PSTL_HOMOGENEOUS_MEMORY__)
  operator const T() const noexcept { return *ptr; }
  device_value<T>& operator=(const T& value) noexcept {
    *ptr = value;
    return *this;
  }
#else
  operator const T() const noexcept {
    return sycl::helpers::read_device_pointer(ptr,
                                              sycl::helpers::default_queue());
  }
  device_value<T>& operator=(const T& value) noexcept {
    sycl::helpers::write_device_pointer(ptr, value,
                                        sycl::helpers::default_queue());
    return *this;
  }
#endif  // __SYCL_DEVICE_ONLY__ or __SYCL_SINGLE_SOURCE__

  device_pointer<T> operator&() const { return device_pointer(ptr); };

  // default copy function isn't what we need.
  device_value<T>& operator=(const device_value<T>& value) noexcept {
    return operator=(T(value));
  }

 private:
  T* ptr;
};

template <typename T>
using device_reference = device_value<T>;

}  // namespace helpers
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_SYCL_DEVICE_VALUE__
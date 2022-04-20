#ifndef __EXPERIMENTAL_DETAIL_SYCL_DEVICE_ITERATOR__
#define __EXPERIMENTAL_DETAIL_SYCL_DEVICE_ITERATOR__

#include <sycl/helpers/sycl_device_pointer.hpp>

namespace sycl {
namespace helpers {

/*
template <typename T>
class device_iterator : public device_pointer<T> {
 public:
  using base = device_pointer<T>;

  device_iterator(T* ptr_, size_t index) : base(ptr_ + index) {}
};
*/

template <typename T>
using device_iterator = device_pointer<T>;

}  // namespace helpers
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_SYCL_DEVICE_ITERATOR__
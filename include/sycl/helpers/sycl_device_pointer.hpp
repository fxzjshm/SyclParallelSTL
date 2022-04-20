#ifndef __EXPERIMENTAL_DETAIL_SYCL_DEVICE_POINTER__
#define __EXPERIMENTAL_DETAIL_SYCL_DEVICE_POINTER__

#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_device_value.hpp>

namespace sycl {
namespace helpers {

// forward declaration as the include order is unknown
template <typename T>
class device_value;

/**
 * @brief A wrapper for pointer that only visible on device. Dereference is
 * forwarded to `device_value`.
 */
// reference: ZipIterator by Dario Pellegrini <pellegrini.dario@gmail.com>,
//            Intel's dpct::device_pointer, boost::compute::buffer_iterator
template <typename T>
class device_pointer {
 public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using pointer = device_pointer<T>;
  using reference = device_value<T>;
  using iterator_category = std::random_access_iterator_tag;

 private:
  /* device */ T *_ptr;

 public:
  device_pointer(T *ptr_, size_type offset = 0) : _ptr(ptr_ + offset) {}

#ifdef __SYCL_DEVICE_ONLY__
  // operator T *() { return _ptr; }
  // operator T *() const { return _ptr; }
#endif  // __SYCL_DEVICE_ONLY__

  device_value<T> operator*() { return device_value<T>(_ptr); }
  device_value<T> operator*() const { return device_value<T>(_ptr); }
  device_value<T> operator[](difference_type rhs) const {
    return *(operator+(rhs));
  }

  inline T *get() const { return _ptr; }

  device_pointer &operator+=(const difference_type d) {
    _ptr += d;
    return *this;
  }
  device_pointer &operator-=(const difference_type d) { return operator+=(-d); }
  device_pointer &operator++() { return operator+=(1); }
  device_pointer &operator--() { return operator+=(-1); }
  device_pointer operator++(int) {
    device_pointer tmp(*this);
    operator++();
    return tmp;
  }
  device_pointer operator--(int) {
    device_pointer tmp(*this);
    operator--();
    return tmp;
  }

  difference_type operator-(const device_pointer &rhs) const {
    return _ptr - rhs._ptr;
  }
  device_pointer operator+(const difference_type d) const {
    device_pointer tmp(*this);
    tmp += d;
    return tmp;
  }
  device_pointer operator-(const difference_type d) const {
    device_pointer tmp(*this);
    tmp -= d;
    return tmp;
  }
  inline friend device_pointer operator+(const difference_type d,
                                         const device_pointer &z) {
    return z + d;
  }
  inline friend device_pointer operator-(const difference_type d,
                                         const device_pointer &z) {
    return z - d;
  }

  bool operator==(const device_pointer &rhs) const { return _ptr == rhs._ptr; }
  bool operator!=(const device_pointer &rhs) const { return _ptr != rhs._ptr; }

  bool operator<=>(device_pointer<T> &other) { return _ptr <=> other._ptr; }
};

template <typename T>
T *get_raw_pointer(const device_pointer<T> &ptr) {
  return ptr.get();
}

template <typename Pointer>
Pointer get_raw_pointer(const Pointer &ptr) {
  return ptr;
}

}  // namespace helpers
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_SYCL_DEVICE_POINTER__
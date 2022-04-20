#ifndef __SYCL_IMPL_CONTAINER_DEVICE_VECTOR__
#define __SYCL_IMPL_CONTAINER_DEVICE_VECTOR__

#include <sycl/algorithm/fill.hpp>
#include <sycl/execution_policy>
#include <sycl/helpers/sycl_default_variables.hpp>
#include <sycl/helpers/sycl_device_allocator.hpp>
#include <sycl/helpers/sycl_device_iterator.hpp>
#include <sycl/helpers/sycl_device_value.hpp>
#include <sycl/helpers/sycl_differences.hpp>

namespace sycl {
namespace impl {

/**
 * @brief A vector that use device memory and device pointers instead of USM
 * ones, may be useful when migrating is expensive if USM is used.
 */
// reference: dpct/del_extras/vector.h, boost/compute/container/vector.hpp
template <typename T, typename Allocator = sycl::helpers::device_allocator<T> >
class device_vector {
 public:
  using iterator = sycl::helpers::device_iterator<T>;
  using const_iterator = const iterator;
  using reference = sycl::helpers::device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = sycl::helpers::device_pointer<T>;
  using const_pointer = const pointer;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

 private:
  Allocator _allocator;
  size_type _size;
  size_type _capacity;
  T *_storage;

  inline constexpr size_type _min_capacity() const { return size_type(1); }
  inline constexpr float _growth_factor() const { return 1.5f; }

 public:
  device_vector(size_type n = 0)
      : _allocator(sycl::helpers::default_queue()),
        _size(n),
        _capacity(
            std::max(static_cast<size_type>(std::ceil(_growth_factor() * n)),
                     _min_capacity())) {
    _storage = _allocator.allocate(_capacity);
  }

  template <typename InputIterator>
  device_vector(InputIterator begin, InputIterator end)
      : device_vector(sycl::helpers::distance(begin, end)) {
    if (size() > 0) {
      if constexpr (std::is_convertible_v<InputIterator, const T *>) {
        sycl::helpers::default_queue().copy(begin, _storage, size()).wait();
      } else {
        ::sycl::impl::copy(sycl::helpers::default_execution_policy(), begin,
                           end, _storage);
      }
    }
  }

  ~device_vector() { _allocator.deallocate(_storage, capacity()); }

  template <typename OtherAllocator>
  device_vector<T, Allocator> &operator=(
      std::vector<T, OtherAllocator> &host_vector) {
    resize(host_vector.size());
    assert(size() == host_vector.size());
    if (size() > 0) {
      sycl::helpers::default_queue()
          .copy(&host_vector[0], _storage, size())
          .wait();
    }
    return *this;
  }

  inline size_type size() const { return _size; }
  inline size_type capacity() const { return _capacity; }

  iterator begin() noexcept { return iterator(_storage, 0); }
  iterator end() noexcept { return iterator(_storage, size()); }
  const_iterator begin() const noexcept { return iterator(_storage, 0); }
  const_iterator end() const noexcept { return iterator(_storage, size()); }
  const_iterator cbegin() const noexcept { return iterator(_storage, 0); }
  const_iterator cend() const noexcept { return iterator(_storage, size()); }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }

  reference operator[](size_type index) const {
    return *(begin() + static_cast<difference_type>(index));
  }

  void clear() { resize(size_type(0)); }

  void reserve(size_type n) {
    if (n > capacity()) {
      size_type old_capacity = capacity();
      size_type new_capacity = _growth_factor() * n;
      auto ptr = _allocator.allocate(new_capacity);
      if (!ptr) throw std::runtime_error("device_vector: Allocation failed");
      sycl::helpers::default_queue().copy(_storage, ptr, size()).wait();
      // use `::sycl::impl::copy(sycl::helpers::default_execution_policy(),
      // begin(), end(), ptr);` if `queue().copy()` not works
      _allocator.deallocate(_storage, old_capacity);
      _storage = ptr;
      _capacity = new_capacity;
    }
  }

  void resize(size_type new_size, const T &x = T()) {
    size_type old_size = size();
    reserve(new_size);
    _size = new_size;
    if (old_size < new_size) {
      ::sycl::impl::fill(sycl::helpers::default_execution_policy(),
                         begin() + old_size, begin() + new_size, x);
    }
  }
};

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_CONTAINER_DEVICE_VECTOR__
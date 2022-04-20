#ifndef __EXPERIMENTAL_DETAIL_SYCL_DEVICE_ALLOCATOR__
#define __EXPERIMENTAL_DETAIL_SYCL_DEVICE_ALLOCATOR__

#include <exception>
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace helpers {

/**
 * @brief An allocator that allocates device memory.
 */
template <typename T, size_t align = sizeof(T)>
class device_allocator {
 public:
  template <typename U, size_t new_align = ((sizeof(T) > 0) ? sizeof(U) : 0)>
  struct rebind {
    typedef device_allocator<U, new_align> other;
  };

  device_allocator(cl::sycl::queue &queue_) : queue(queue_){};

  T *allocate(std::size_t num_elements) {
    T *ptr = cl::sycl::aligned_alloc_device<T>(align, num_elements, queue);
    if (!ptr) throw std::runtime_error("device_allocator: Allocation failed");
    return ptr;
  }

  void deallocate(T *ptr, std::size_t size) {
    if (ptr) sycl::free(ptr, queue);
  }

 private:
  cl::sycl::queue queue;
};

}  // namespace helpers
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_SYCL_DEVICE_ALLOCATOR__
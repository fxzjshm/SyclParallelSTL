#ifndef __EXPERIMENTAL_DETAIL_SYCL_USM_VECTOR__
#define __EXPERIMENTAL_DETAIL_SYCL_USM_VECTOR__

#include <sycl/helpers/sycl_default_variables.hpp>
#include <vector>

namespace sycl {
namespace helpers {

template <typename T,
          cl::sycl::usm::alloc AllocKind = cl::sycl::usm::alloc::shared,
          size_t align = sizeof(T)>
class usm_allocator : public cl::sycl::usm_allocator<T, AllocKind, align> {
 public:
  template <typename U, size_t new_align = ((sizeof(T) > 0) ? sizeof(U) : 0)>
  struct rebind {
    typedef usm_allocator<U, AllocKind, new_align> other;
  };

  usm_allocator(const cl::sycl::queue& q = sycl::helpers::default_queue())
      : cl::sycl::usm_allocator<T, AllocKind, align>(q){};
};

template <typename T,
          cl::sycl::usm::alloc AllocKind = cl::sycl::usm::alloc::shared,
          size_t align = sizeof(T)>
using usm_vector = std::vector<T, usm_allocator<T, AllocKind, align>>;

}  // namespace helpers
}  // namespace sycl

#endif
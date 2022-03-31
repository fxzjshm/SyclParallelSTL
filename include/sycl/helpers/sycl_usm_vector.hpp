#ifndef __EXPERIMENTAL_DETAIL_SYCL_USM_VECTOR__
#define __EXPERIMENTAL_DETAIL_SYCL_USM_VECTOR__

#include <CL/sycl.hpp>
#include <vector>

namespace sycl {
namespace helpers {

template <typename T,
          cl::sycl::usm::alloc AllocKind = cl::sycl::usm::alloc::shared,
          size_t align = sizeof(T)>
class usm_alloc : public cl::sycl::usm_allocator<T, AllocKind, align> {
 public:
  template <typename U>
  struct rebind {
    typedef usm_alloc<U, AllocKind, sizeof(U)> other;
  };

  usm_alloc()
      : cl::sycl::usm_allocator<T, AllocKind, align>(cl::sycl::queue()){};
};

template <typename T,
          cl::sycl::usm::alloc AllocKind = cl::sycl::usm::alloc::shared,
          size_t align = sizeof(T)>
using usm_vector = std::vector<T, usm_alloc<T, AllocKind, align>>;

}  // namespace helpers
}  // namespace sycl

#endif
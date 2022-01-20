#ifndef __SYCL_IMPL_ALGORITHM_COPY__
#define __SYCL_IMPL_ALGORITHM_COPY__

#include <type_traits>
#include <algorithm>
#include <iostream>

#include <sycl/algorithm/transform.hpp>

namespace sycl {
namespace impl {

template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
OutputIterator copy(
    ExecutionPolicy& snp, InputIterator first, InputIterator last, OutputIterator result) {
  return ::sycl::impl::transform(snp, first, last, result, std::identity());
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_COPY__

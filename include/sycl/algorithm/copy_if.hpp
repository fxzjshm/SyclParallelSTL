#ifndef __SYCL_IMPL_ALGORITHM_COPY_IF__
#define __SYCL_IMPL_ALGORITHM_COPY_IF__

#include <type_traits>
#include <algorithm>
#include <iostream>

#include <sycl/algorithm/transform_if.hpp>

namespace sycl {
namespace impl {

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator, typename Predicate>
OutputIterator copy_if(
    ExecutionPolicy& snp, InputIterator first, InputIterator last,
    OutputIterator result, Predicate predicate) {

  return ::sycl::impl::copy_if(snp, first, last, first, result, predicate);
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename Predicate>
OutputIterator copy_if(
    ExecutionPolicy& snp, InputIterator1 first, InputIterator1 last,
    InputIterator2 stencil, OutputIterator result, Predicate predicate) {

  return ::sycl::impl::transform_if(snp, first, last, stencil, result, std::identity(), predicate);
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_COPY_IF__

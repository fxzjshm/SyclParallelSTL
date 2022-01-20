#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_SCATTER_IF__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_SCATTER_IF__

#include <type_traits>
#include <algorithm>
#include <iostream>

// Detail header
#include <sycl/helpers/sycl_buffers.hpp>

#include <sycl/algorithm/transform_if.hpp>
#include <boost/iterator/permutation_iterator.hpp>

namespace sycl {
namespace impl {

template <class ExecutionPolicy, class InputIterator, class MapIterator,
          class StencilIterator, class OutputIterator, class Predicate>
void scatter_if(ExecutionPolicy &exec, InputIterator first, InputIterator last,
                MapIterator map, StencilIterator stencil, OutputIterator result,
                Predicate predicate) {
  sycl::impl::transform_if(exec, first, last, stencil, boost::iterators::make_permutation_iterator(result, map), std::identity(), predicate);
}

template <class ExecutionPolicy, class InputIterator, class MapIterator,
          class StencilIterator, class OutputIterator>
void scatter_if(ExecutionPolicy &exec, InputIterator first, InputIterator last,
                MapIterator map, StencilIterator stencil, OutputIterator result) {
  sycl::impl::scatter_if(exec, first, last, map, stencil, result, std::identity());
}

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_SCATTER_IF__

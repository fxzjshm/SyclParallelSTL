#ifndef __SYCL_IMPL_ALGORITHM_COPY_IF__
#define __SYCL_IMPL_ALGORITHM_COPY_IF__

#include <type_traits>
#include <algorithm>
#include <iostream>

#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename Predicate>
OutputIterator copy_if(
    ExecutionPolicy& exec, InputIterator1 first, InputIterator1 last,
    InputIterator2 stencil, OutputIterator result, Predicate predicate) {

  auto count = std::distance(first, last);
  if (count == 0) {
      return result;
  }

  cl::sycl::queue q(exec.get_queue());
  auto device = q.get_device();

  // reference: boost/compute/algorithm/transform_if.cpp
  size_t* indices = sycl::helpers::make_temp_device_pointer<size_t, 0>(count, q);
  ::sycl::impl::transform(exec, stencil, stencil + count, indices,
    [predicate](auto x){
      return (size_t)((predicate(x)) ? 1 : 0);
  });

  size_t copied_element_count = sycl::helpers::read_device_pointer(indices + count - 1, q);
  ::sycl::impl::exclusive_scan(
      exec, indices, indices + count, indices, (size_t)0, std::plus()
  );
  copied_element_count += sycl::helpers::read_device_pointer(indices + count - 1, q); // last scan element plus last mask element
  assert(copied_element_count <= count);
  if (copied_element_count == 0) {
      return result;
  }

  auto vectorSize = count;
  const auto ndRange = exec.calculateNdRange(vectorSize);
  auto transform_if_do_copy = [vectorSize, ndRange, first, indices, stencil, result, predicate, copied_element_count] (cl::sycl::handler &h) {
    auto aI = first;
    auto aIndices = indices;
    auto aStencil = stencil;
    auto aO = result;
    h.parallel_for(
        ndRange, [vectorSize, aI, aIndices, aStencil, aO, predicate, copied_element_count](cl::sycl::nd_item<1> id) {
          if (id.get_global_id(0) < vectorSize) {
            if (predicate(aStencil[id.get_global_id(0)])) {
              //auto idx = aIndices[id.get_global_id(0)];
              //assert(idx < copied_element_count);
              aO[aIndices[id.get_global_id(0)]] = aI[id.get_global_id(0)];
            }
          }
        });
  };
  q.submit(transform_if_do_copy).wait();
  return result + copied_element_count;
}

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator, typename Predicate>
OutputIterator copy_if(
    ExecutionPolicy& exec, InputIterator first, InputIterator last,
    OutputIterator result, Predicate predicate) {

  return ::sycl::impl::copy_if(exec, first, last, first, result, predicate);
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_COPY_IF__

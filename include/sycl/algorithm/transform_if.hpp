#ifndef __SYCL_IMPL_ALGORITHM_TRANSFORM_IF__
#define __SYCL_IMPL_ALGORITHM_TRANSFORM_IF__

#include <type_traits>
#include <algorithm>
#include <iostream>

#include <sycl/execution_policy>
#include <experimental/execution_policy>
#include <experimental/algorithm>
#include <sycl/algorithm/transform.hpp>
#include <sycl/algorithm/exclusive_scan.hpp>

namespace sycl {
namespace impl {

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator, typename UnaryFunction, typename Predicate>
OutputIterator transform_if(
    ExecutionPolicy& exec, InputIterator first, InputIterator last,
    OutputIterator result, UnaryFunction function, Predicate predicate) {
  return ::sycl::impl::transform_if(snp, first, last, first, result, function, predicate);
}

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename UnaryFunction, typename Predicate>
OutputIterator transform_if(
    ExecutionPolicy& exec, InputIterator1 first, InputIterator1 last, InputIterator2 stencil,
    OutputIterator result, UnaryFunction function, Predicate predicate) {

  cl::sycl::queue q(exec.get_queue());
  auto device = q.get_device();
  auto bufI = sycl::helpers::make_const_buffer(first, last);
  auto count = bufI.get_count();
  auto bufO = sycl::helpers::make_buffer(result, result + bufI.get_count());

  // reference: boost/compute/algorithm/transform_if.cpp
  auto indices = std::vector<uint8_t>(count), sums = std::vector<uint8_t>(count);
  ::sycl::impl::transform(exec, stencil, stencil + count, indices.begin(),
    [predicate](auto x){
      return (uint8_t)((predicate(x)) ? 1 : 0);
  });

  size_t copied_element_count = *(indices.cend() - 1);
  ::std::experimental::parallel::exclusive_scan(
      exec, indices.begin(), indices.end(), sums.begin(), (uint8_t)0
  );
  copied_element_count += *(sums.cend() - 1); // last scan element plus last mask element

  auto vectorSize = count;
  const auto ndRange = exec.calculateNdRange(vectorSize);
  auto bufSums = sycl::helpers::make_buffer(sums.begin(), sums.end());
  auto transform_if_do_copy = [vectorSize, ndRange, &bufI, &bufSums, &bufO, predicate, function](
      cl::sycl::handler &h) {
    auto aI = bufI.template get_access<cl::sycl::access::mode::read>(h);
    auto aSums = bufSums.template get_access<cl::sycl::access::mode::read>(h);
    auto aO = bufO.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for(
        ndRange, [aI, aSums, aO, predicate, function](cl::sycl::nd_item<1> id) {
          if (predicate(aI[id.get_global_id(0)])) {
            aO[aSums[id.get_global_id(0)]] = function(aI[id.get_global_id(0)]);
          }
        });
  };
  q.submit(transform_if_do_copy);
  return result + copied_element_count;
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_TRANSFORM_IF__

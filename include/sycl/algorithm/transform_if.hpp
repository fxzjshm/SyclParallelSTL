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
  return ::sycl::impl::transform_if(exec, first, last, first, result, function, predicate);
}

// NOTICE: transform_if in Thrust will not modify output if stencil == false,
//   while transform_if in Boost.Compute seems to delete all stencil == false values in output.
// here the former one is chosen, the later should now be accomplished by a copy_if + transform
#if 1
template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename UnaryFunction, typename Predicate>
OutputIterator transform_if(
    ExecutionPolicy& exec, InputIterator1 first, InputIterator1 last, InputIterator2 stencil,
    OutputIterator result, UnaryFunction function, Predicate predicate) {

  auto count = std::distance(first, last);
  if (count == 0) {
      return result;
  }

  cl::sycl::queue q(exec.get_queue());
  auto device = q.get_device();

  auto vectorSize = count;
  const auto ndRange = exec.calculateNdRange(vectorSize);
  auto transform_if_do_copy = [vectorSize, ndRange, first, stencil, result, predicate, function](cl::sycl::handler &h) {
    h.parallel_for(
        ndRange, [first, stencil, result, predicate, function](cl::sycl::nd_item<1> id) {
          if (predicate(stencil[id.get_global_id(0)])) {
            result[id.get_global_id(0)] = function(first[id.get_global_id(0)]);
          }
        }
    );
  };
  q.submit(transform_if_do_copy).wait();
  return result + count;
}

#else

template <typename ExecutionPolicy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename UnaryFunction, typename Predicate>
OutputIterator transform_if(
    ExecutionPolicy& exec, InputIterator1 first, InputIterator1 last, InputIterator2 stencil,
    OutputIterator result, UnaryFunction function, Predicate predicate) {

  auto count = std::distance(first, last);
  if (count == 0) {
      return result;
  }

  cl::sycl::queue q(exec.get_queue());
  auto device = q.get_device();
  auto bufI = sycl::helpers::make_const_buffer(first, last);
  assert(bufI.get_count() == count);

  auto bufStencil = sycl::helpers::make_const_buffer(stencil, stencil + count);

  // reference: boost/compute/algorithm/transform_if.cpp
  auto indices = std::vector<size_t>(count);
  ::sycl::impl::transform(exec, stencil, stencil + count, indices.begin(),
    [predicate](auto x){
      return (size_t)((predicate(x)) ? 1 : 0);
  });

  size_t copied_element_count = *(indices.cend() - 1);
  ::sycl::impl::exclusive_scan(
      exec, indices.begin(), indices.end(), indices.begin(), (size_t)0, std::plus()
  );
  copied_element_count += *(indices.cend() - 1); // last scan element plus last mask element
  if (copied_element_count == 0) {
      return result;
  }
  auto bufO = sycl::helpers::make_buffer(result, result + copied_element_count);

  auto vectorSize = count;
  const auto ndRange = exec.calculateNdRange(vectorSize);
  auto bufIndices = sycl::helpers::make_const_buffer(indices.begin(), indices.end());
  auto transform_if_do_copy = [vectorSize, ndRange, &bufI, &bufIndices, &bufStencil, &bufO, predicate, function, copied_element_count](
      cl::sycl::handler &h) {
    auto aI = bufI.template get_access<cl::sycl::access::mode::read>(h);
    auto aIndices = bufIndices.template get_access<cl::sycl::access::mode::read>(h);
    auto aStencil = bufStencil.template get_access<cl::sycl::access::mode::read>(h);
    auto aO = bufO.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for(
        ndRange, [aI, aIndices, aStencil, aO, predicate, function, copied_element_count](cl::sycl::nd_item<1> id) {
          if (predicate(aStencil[id.get_global_id(0)])) {
            auto idx = aIndices[id.get_global_id(0)];
            assert(idx < copied_element_count);
            aO[aIndices[id.get_global_id(0)]] = function(aI[id.get_global_id(0)]);
          }
        });
  };
  q.submit(transform_if_do_copy);
  return result + copied_element_count;
}

#endif // different understanding of transform_if

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_TRANSFORM_IF__

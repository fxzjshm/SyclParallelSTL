#ifndef __SYCL_IMPL_ALGORITHM_IOTA__
#define __SYCL_IMPL_ALGORITHM_IOTA__

#include <type_traits>
#include <algorithm>
#include <iostream>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>

// modified from fill

namespace sycl {
namespace impl {

/* iota.
 * Implementation of the command group that submits a iota kernel.
 * The kernel is implemented as a lambda.
 */
template <typename ExecutionPolicy, typename ForwardIt, typename T = typename std::iterator_traits<ForwardIt>::value_type>
void iota(ExecutionPolicy &sep, ForwardIt b, ForwardIt e, const T &value = T(0)) {
  cl::sycl::queue q { sep.get_queue() };
  auto device = q.get_device();
  auto bufI = helpers::make_buffer( b, e );
  // copy value into a local variable, as we cannot capture it by reference
  T val = value;
  auto vectorSize = bufI.get_count();
  const auto ndRange = sep.calculateNdRange(vectorSize);
  auto f = [vectorSize, ndRange, &bufI, val](
      cl::sycl::handler &h) mutable {
    auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<typename ExecutionPolicy::kernelName>(
        ndRange, [aI, val, vectorSize](cl::sycl::nd_item<1> id) {
          if (id.get_global_id(0) < vectorSize) {
            aI[id.get_global_id(0)] = val + id.get_global_id(0);
          }
        });
  };
  q.submit(f);
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_IOTA__

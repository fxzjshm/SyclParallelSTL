#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_GATHER__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_GATHER__

#include <type_traits>
#include <algorithm>
#include <iostream>

// Detail header
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

template <class ExecutionPolicy, class MapIterator, class InputIterator,
          class OutputIterator>
void gather(ExecutionPolicy &sep, MapIterator first, MapIterator last,
            InputIterator input, OutputIterator result) {
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();
  auto map = sycl::helpers::make_const_buffer(first, last);
  auto n = map.get_count();
  auto in = sycl::helpers::make_const_buffer(input, input + n);
  auto res = sycl::helpers::make_buffer(result, result + n);
  const auto ndRange = sep.calculateNdRange(n);
  auto f = [n, ndRange, &map, &in, &res](
      cl::sycl::handler &h) mutable {
    auto a_map = map.template get_access<cl::sycl::access::mode::read>(h);
    auto a_in = in.template get_access<cl::sycl::access::mode::read>(h);
    auto a_res = res.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for(
        ndRange, [a_map, a_in, a_res, n](cl::sycl::nd_item<1> id) {
          if (id.get_global_id(0) < n) {
            a_res[id.get_global_id(0)] = a_in[a_map[id.get_global_id(0)]];
          }
        });
  };
  q.submit(f);
}

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_GATHER__

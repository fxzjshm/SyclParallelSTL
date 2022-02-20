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
  auto n = std::distance(first, last);
  const auto ndRange = sep.calculateNdRange(n);
  auto f = [n, ndRange, first, input, result] (cl::sycl::handler &h) mutable {
    auto a_map = first;
    auto a_in = input;
    auto a_res = result;
    h.parallel_for(
        ndRange, [a_map, a_in, a_res, n](cl::sycl::nd_item<1> id) {
          if (id.get_global_id(0) < n) {
            a_res[id.get_global_id(0)] = a_in[a_map[id.get_global_id(0)]];
          }
        });
  };
  q.submit(f).wait();
}

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_GATHER__

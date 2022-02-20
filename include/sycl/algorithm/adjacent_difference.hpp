#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_ADJACENT_DIFFERENCE__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_ADJACENT_DIFFERENCE__

#include <type_traits>
#include <algorithm>
#include <iostream>

// Detail header
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

template <class ExecutionPolicy, class Iterator, class OutputIterator,
          class BinaryOperation>
OutputIterator adjacent_difference(ExecutionPolicy &sep, Iterator b, Iterator e,
                                   OutputIterator out, BinaryOperation op) {
  {
    auto n = std::distance(b, e);
    if (n == 0) {
      return out;
    }

    cl::sycl::queue q(sep.get_queue());
    auto device = q.get_device();
    auto vectorSize = n;
    const auto ndRange = sep.calculateNdRange(vectorSize);
    auto f = [vectorSize, ndRange, b, out, op] (cl::sycl::handler &h) {
      auto aI = b;
      auto aO = out;
      h.parallel_for(
          ndRange, [aI, aO, op, vectorSize](cl::sycl::nd_item<1> id) {
            auto i = id.get_global_id(0);
            if ((i < vectorSize)) {
              if (i == 0) {
                aO[0] = aI[0];
              } else {
                aO[i] = op(aI[i], aI[i-1]);
              }
            }
          });
    };
    q.submit(f).wait();
    return out + n;
  }
}

template <class ExecutionPolicy, class Iterator, class OutputIterator>
OutputIterator adjacent_difference(ExecutionPolicy &sep, Iterator b, Iterator e, OutputIterator out) {
  return sycl::impl::adjacent_difference(sep, b, e, out, std::minus());
}

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_ADJACENT_DIFFERENCE__

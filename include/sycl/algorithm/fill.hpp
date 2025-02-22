/* Copyright (c) 2015-2018 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

*/

#ifndef __SYCL_IMPL_ALGORITHM_FILL__
#define __SYCL_IMPL_ALGORITHM_FILL__

#include <type_traits>
#include <algorithm>
#include <iostream>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

/* fill.
 * Implementation of the command group that submits a fill kernel.
 * The kernel is implemented as a lambda.
 */
template <typename ExecutionPolicy, typename ForwardIt, typename T>
void fill(ExecutionPolicy &sep, ForwardIt b, ForwardIt e, const T &value) {
  cl::sycl::queue q { sep.get_queue() };
  auto device = q.get_device();
  // copy value into a local variable, as we cannot capture it by reference
  T val = value;
  auto vectorSize = std::distance(b, e);
  const auto ndRange = sep.calculateNdRange(vectorSize);
  auto f = [vectorSize, ndRange, b, val] (cl::sycl::handler &h) mutable {
    auto aI = b;
    h.parallel_for(
        ndRange, [aI, val, vectorSize](cl::sycl::nd_item<1> id) {
          if (id.get_global_id(0) < vectorSize) {
            aI[id.get_global_id(0)] = val;
          }
        });
  };
  q.submit(f).wait();
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_FILL__

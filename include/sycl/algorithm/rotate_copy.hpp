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

#ifndef __SYCL_IMPL_ALGORITHM_ROTATE_COPY__
#define __SYCL_IMPL_ALGORITHM_ROTATE_COPY__

#include <algorithm>
#include <vector>

namespace sycl {
namespace impl {

/* rotate_copy.
 * Implementation of the command group that submits a rotate_copy kernel.
 */
template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
ForwardIt2 rotate_copy(ExecutionPolicy &sep, ForwardIt1 first,
                       ForwardIt1 middle, ForwardIt1 last,
                       ForwardIt2 result) {

  if (first == last) return result;

  const size_t n = std::distance(first, last);

  using namespace cl::sycl;
  using value_type = typename std::iterator_traits<ForwardIt1>::value_type;
{
  const auto rot_n = std::distance(first, middle);
  sep.get_queue().submit([n, rot_n, first, result](handler &h) {

    auto aI = first;
    auto aO = result;

    h.parallel_for(range<1>{n}, [aI, aO, rot_n](item<1> i) {
      const size_t rotated_id = (i.get_id(0) + rot_n >= i.get_range(0)) ?
                                 i.get_id(0) + rot_n  - i.get_range(0)  :
                                 i.get_id(0) + rot_n;
      aO[i.get_id(0)] = aI[rotated_id];
    });
  }).wait();
}
  return std::next(result, n);
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_ROTATE_COPY__

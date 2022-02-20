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

#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__

#include <type_traits>
#include <algorithm>
#include <iostream>

// Detail header
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

/** transform sycl implementation
 * @brief Function that takes a Unary Operator and applies to the given range
 * @param sep : Execution Policy
 * @param b   : Start of the range
 * @param e   : End of the range
 * @param out : Output iterator
 * @param op  : Unary Operator
 * @return  An iterator pointing to the last element
 */
template <class ExecutionPolicy, class Iterator, class OutputIterator,
          class UnaryOperation>
OutputIterator transform(ExecutionPolicy &sep, Iterator b, Iterator e,
                         OutputIterator out, UnaryOperation op) {
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
      h.parallel_for(
          ndRange, [b, out, op, vectorSize](cl::sycl::nd_item<1> id) {
            if ((id.get_global_id(0) < vectorSize)) {
              out[id.get_global_id(0)] = op(b[id.get_global_id(0)]);
            }
          });
    };
    q.submit(f).wait();
    return out + n;
  }
}

/** transform sycl implementation
* @brief Function that takes a Binary Operator and applies to the given range
* @param sep    : Execution Policy
* @param first1 : Start of the range of buffer 1
* @param last1  : End of the range of buffer 1
* @param first2 : Start of the range of buffer 2
* @param result : Output iterator
* @param op     : Binary Operator
* @return  An iterator pointing to the last element
*/
template <class ExecutionPolicy, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryOperation>
OutputIterator transform(ExecutionPolicy &sep, InputIterator1 first1,
                         InputIterator1 last1, InputIterator2 first2,
                         OutputIterator result, BinaryOperation op) {
  auto n = std::distance(first1, last1);
  if (n == 0) {
      return result;
  }
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();
  const auto ndRange = sep.calculateNdRange(n);
  auto f = [n, ndRange, first1, first2, result, op] (cl::sycl::handler &h) mutable {
    h.parallel_for(
        ndRange, [first1, first2, result, op, n](cl::sycl::nd_item<1> id) {
          if (id.get_global_id(0) < n) {
            result[id.get_global_id(0)] = op(first1[id.get_global_id(0)], first2[id.get_global_id(0)]);
          }
        });
  };
  q.submit(f).wait();
  return result + n;
}

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__

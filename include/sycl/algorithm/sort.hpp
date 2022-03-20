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

#ifndef __SYCL_IMPL_ALGORITHM_SORT__
#define __SYCL_IMPL_ALGORITHM_SORT__

#include <type_traits>
#include <typeinfo>
#include <algorithm>

#include <sycl/algorithm/copy.hpp>
#include <ZipIterator.hpp>

/** sort_kernel_bitonic.
 * Class used to name the bitonic kernel sort per type.
 */
template <typename T>
class sort_kernel_bitonic;

/** sequential_sort_name.
 * Class used to name the bitonic kernel sort per type.
 */
template <typename T>
class sequential_sort_name {
  T userGivenKernelName;
};

/** bitonic_sort_name.
 * Class used to name the bitonic kernel sort per type.
 */
template <typename T>
class bitonic_sort_name {
  T userGivenKernelName;
};

/* sort_swap.
 * Basic simple swap used inside the sort functions.
 */
template <typename T>
void sort_swap(T &lhs, T &rhs) {
  auto temp = rhs;
  rhs = lhs;
  lhs = temp;
}

/* sort_kernel_sequential.
 * Simple kernel to sequentially sort a vector
 */
template <typename T>
class sort_kernel_sequential {
  /* Aliases for SYCL accessors */
  using sycl_rw_acc =
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;

  sycl_rw_acc a_;
  size_t vS_;

 public:
  sort_kernel_sequential(sycl_rw_acc a, size_t vectorSize)
      : a_(a), vS_(vectorSize){};

  // Simple sequential sort
  void operator()() {
    for (size_t i = 0; i < vS_; i++) {
      for (size_t j = 1; j < vS_; j++) {
        if (a_[j - 1] > a_[j]) {
          sort_swap<T>(a_[j - 1], a_[j]);
        }
      }
    }
  }
};  // class sort_kernel

/* sort_kernel_sequential.
 * Simple kernel to sequentially sort a vector
 */
template <typename T, class ComparableOperator>
class sort_kernel_sequential_comp {
  /* Aliases for SYCL accessors */
  using sycl_rw_acc =
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;

  sycl_rw_acc a_;
  size_t vS_;
  ComparableOperator comp_;

 public:
  sort_kernel_sequential_comp(sycl_rw_acc a, size_t vectorSize,
                              ComparableOperator comp)
      : a_(a), vS_(vectorSize), comp_(comp){};

  // Simple sequential sort
  void operator()() const {
    for (size_t i = 0; i < vS_; i++) {
      for (size_t j = 1; j < vS_; j++) {
        if (!comp_(a_[j - 1], a_[j])) {
          sort_swap<T>(a_[j - 1], a_[j]);
        }
      }
    }
  }
};  // class sort_kernel

namespace sycl {
namespace impl {

/* Aliases for SYCL accessors */
template <typename T>
using sycl_rw_acc = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                                       cl::sycl::access::target::global_buffer>;

/** isPowerOfTwo.
 * Quick check to ensure num is a power of two.
 * Will only work with integers.
 * @return true if num is power of two
 */
template <typename T>
inline bool isPowerOfTwo(T num) {
  return (num != 0) && !(num & (num - 1));
}

template <>
inline bool isPowerOfTwo<float>(float num) = delete;
template <>
inline bool isPowerOfTwo<double>(double num) = delete;

/** sequential_sort.
 * Command group to call the sequential sort kernel */
template <typename T, typename Alloc>
void sequential_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1, Alloc> buf,
                     size_t vectorSize) {
  auto f = [buf, vectorSize](cl::sycl::handler &h) mutable {
    auto a = buf.template get_access<cl::sycl::access::mode::read_write>(h);
    h.single_task(sort_kernel_sequential<T>(a, vectorSize));
  };
  q.submit(f);
}

/** sequential_sort.
 * Command group to call the sequential sort kernel */
template <typename T, typename Alloc, class ComparableOperator>
void sequential_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1, Alloc> buf,
                     size_t vectorSize, ComparableOperator comp) {
  auto f = [buf, vectorSize, comp](cl::sycl::handler &h) mutable {
    auto a = buf.template get_access<cl::sycl::access::mode::read_write>(h);
    h.single_task(sort_kernel_sequential_comp<T, ComparableOperator>(
        a, vectorSize, comp));
  };
  q.submit(f);
}

// reference: boost/compute/algorithm/detail/merge_sort_on_gpu.hpp
template<class ExecutionPolicy, class InputIterator, class OutputIterator, class Compare>
inline void merge_blocks_on_gpu(ExecutionPolicy &exec,
                                InputIterator input_iterator,
                                OutputIterator output_iterator,
                                Compare compare,
                                const size_t count,
                                const size_t block_size)
{
    typedef typename std::iterator_traits<InputIterator>::value_type key_type;

    const auto ndRange = exec.calculateNdRange(count);

    auto f = [=] (cl::sycl::handler &h) {
        auto input = input_iterator;
        auto output = output_iterator;
        h.parallel_for(ndRange, [=](cl::sycl::nd_item<1> id) {
            const size_t gid = id.get_global_id(0);
            if (gid >= count) {
                return;
            }
            const key_type my_key = input[gid];
            const size_t my_block_idx = gid / block_size;
            const bool my_block_idx_is_odd = my_block_idx & 0x1;
            const size_t other_block_idx = my_block_idx_is_odd ? my_block_idx - 1 : my_block_idx + 1;
            const size_t my_block_start = std::min(my_block_idx * block_size, count);
            const size_t my_block_end = std::min((my_block_idx + 1) * block_size, count);
            const size_t other_block_start = std::min(other_block_idx * block_size, count);
            const size_t other_block_end = std::min((other_block_idx + 1) * block_size, count);
            if (other_block_start == count) {
                output[gid] = my_key;
                return;
            }
            size_t left_idx = other_block_start;
            size_t right_idx = other_block_end;
            while (left_idx < right_idx) {
                size_t mid_idx = (left_idx + right_idx) / 2;
                key_type mid_key = input[mid_idx];
                bool smaller = compare(mid_key, my_key);
                left_idx = smaller ? mid_idx + 1 : left_idx;
                right_idx = smaller ? right_idx : mid_idx;
            }
            right_idx = other_block_end;
            if (my_block_idx_is_odd && left_idx != right_idx) {
                key_type upper_key = input[left_idx];
                while (!(compare(upper_key, my_key)) && !(compare(my_key, upper_key)) && left_idx < right_idx) {
                    size_t mid_idx = (left_idx + right_idx) / 2;
                    key_type mid_key = input[mid_idx];
                    bool equal = !(compare(mid_key, my_key)) && !(compare(my_key, mid_key));
                    left_idx = equal ? mid_idx + 1 : left_idx + 1;
                    right_idx = equal ? right_idx : mid_idx;
                    upper_key = input[left_idx];
                }
            }
            size_t offset = 0;
            offset += gid - my_block_start;
            offset += left_idx - other_block_start;
            offset += std::min(my_block_start, other_block_start);
            output[offset] = my_key;
        });
    };
    exec.get_queue().submit(f).wait();
}

// reference: boost/compute/algorithm/detail/merge_sort_on_gpu.hpp
template<class ExecutionPolicy, class Iterator, class Compare>
inline void merge_sort_on_gpu(ExecutionPolicy &exec,
                              Iterator first,
                              Iterator last,
                              Compare compare)
{
    typedef typename std::iterator_traits<Iterator>::value_type key_type;

    size_t count = std::distance(first, last);
    if(count < 2){
        return;
    }

    cl::sycl::queue queue = exec.get_queue();
    size_t block_size = 1;

    bool result_in_temporary_buffer = false;
    key_type* temp_keys = sycl::helpers::make_temp_device_pointer<key_type>(count, queue);

    for(; block_size < count; block_size *= 2) {
        result_in_temporary_buffer = !result_in_temporary_buffer;
        if(result_in_temporary_buffer) {
            merge_blocks_on_gpu(exec, first, temp_keys,
                                compare, count, block_size);
        } else {
            merge_blocks_on_gpu(exec, temp_keys, first,
                                compare, count, block_size);
        }
    }

    if(result_in_temporary_buffer) {
        ::sycl::impl::copy(exec, temp_keys, temp_keys + count, first);
    }
    queue.wait();
}

/* bitonic_sort.
 * Performs a bitonic sort on the given buffer
 */
template <typename InputIterator>
void bitonic_sort(cl::sycl::queue q, InputIterator input,
                  size_t vectorSize) {
  typedef typename std::iterator_traits<InputIterator>::value_type T;
  ::sycl::impl::bitonic_sort(q, input, vectorSize, std::less<T>());
}  // bitonic_sort

/* bitonic_sort.
 * Performs a bitonic sort on the given buffer
 */
template <typename InputIterator, class ComparableOperator>
void bitonic_sort(cl::sycl::queue q, InputIterator input,
                  size_t vectorSize, ComparableOperator comp) {
  typedef typename std::iterator_traits<InputIterator>::value_type T;

  int numStages = 0;
  // 2^numStages should be equal to length
  // i.e number of times you halve the lenght to get 1 should be numStages
  for (int tmp = vectorSize; tmp > 1; tmp >>= 1) {
    ++numStages;
  }
  cl::sycl::range<1> r{vectorSize / 2};
  for (int stage = 0; stage < numStages; ++stage) {
    // Every stage has stage + 1 passes
    for (int passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
      auto f = [=](cl::sycl::handler &h) mutable {
        auto a = input;
        h.parallel_for(
            cl::sycl::range<1>{r},
            [a, stage, passOfStage, comp](cl::sycl::item<1> it) {
              int sortIncreasing = 1;
              cl::sycl::id<1> id = it.get_id();
              int threadId = id.get(0);

              int pairDistance = 1 << (stage - passOfStage);
              int blockWidth = 2 * pairDistance;

              int leftId = (threadId % pairDistance) +
                           (threadId / pairDistance) * blockWidth;
              int rightId = leftId + pairDistance;

              T leftElement = a[leftId];
              T rightElement = a[rightId];

              int sameDirectionBlockWidth = 1 << stage;

              if ((threadId / sameDirectionBlockWidth) % 2 == 1) {
                sortIncreasing = 1 - sortIncreasing;
              }

              T greater;
              T lesser;

              if (comp(leftElement, rightElement)) {
                greater = rightElement;
                lesser = leftElement;
              } else {
                greater = leftElement;
                lesser = rightElement;
              }

              a[leftId] = sortIncreasing ? lesser : greater;
              a[rightId] = sortIncreasing ? greater : lesser;
            });
      };  // command group functor
      q.submit(f).wait();
    }  // passStage
  }    // stage
}  // bitonic_sort

template<typename T>
struct buffer_traits;

template<typename T, typename Alloc>
struct buffer_traits<cl::sycl::buffer<T, 1, Alloc>> {
  typedef Alloc allocator_type;
};

/** sort
 * @brief Function that takes a Comp Operator and applies it to the given range
 * @param sep   : Execution Policy
 * @param first : Start of the range
 * @param last  : End of the range
 * @param comp  : Comp Operator
 */
template <class ExecutionPolicy, class RandomIt, class CompareOp>
void sort(ExecutionPolicy &sep, RandomIt first, RandomIt last, CompareOp comp) {
  cl::sycl::queue q(sep.get_queue());
  auto vectorSize = std::distance(first, last);
  
  if (impl::isPowerOfTwo(vectorSize)) {
    sycl::impl::bitonic_sort<
        RandomIt, CompareOp>(
        q, first, vectorSize, comp);
  } else {
    sycl::impl::merge_sort_on_gpu<
        ExecutionPolicy, RandomIt, CompareOp>(
        sep, first, last, comp);
  }
  q.wait();
}

template <class ExecutionPolicy, class KeyItrator, class ValueItrator, class CompareOp>
void sort_by_key(ExecutionPolicy &sep, KeyItrator key_first, KeyItrator key_last,
                 ValueItrator value_first, CompareOp comp) {
  auto n = std::distance(key_first, key_last);
  sort(sep, ZipIter(key_first, value_first), ZipIter(key_first, value_first) + n,
    [comp] (auto a, auto b) {
      return comp(std::get<0>(a), std::get<0>(b));
    }
  );
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_SORT__

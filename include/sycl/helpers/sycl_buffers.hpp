/*
 * Copyright (c) 2015-2018 The Khronos Group Inc.

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

/**
 * @file
 * @brief Helper functions to create buffers from iterators
 * @detail This file contains all functions that enable the creation of buffers
 *    (or images) from Iterators. The make_XXX functions that take two
 * parameters
 *    perform a dispatch depending on the iterator tag to call the appropriate
 *    call depending on the iterator tag used.
 *    Enable-if is used to differentiate the case of SYCL-enabled iterators from
 *    the standard STL iterators.
 */

#ifndef __EXPERIMENTAL_DETAIL_SYCL_BUFFERS__
#define __EXPERIMENTAL_DETAIL_SYCL_BUFFERS__

#include <type_traits>
#include <typeinfo>
#include <memory>

/** @defgroup sycl_helpers
 *
 * Contains definitions of classes and functions that help
 * dealing with the SYCL interface.
 *
 */

#include <sycl/helpers/sycl_iterator.hpp>

/** \addtogroup sycl
 * @{
 */
namespace sycl {
namespace helpers {

/**
 *
 * @brief Creates a buffer from a random access iterator that triggers
 *  a copy back operation.
 * @param Iterator b  Start of the range
 * @param Iterator e  End of the range
 * @param std::random_access_iterator_tag Used for iterator dispatch only
 */
template <typename Iterator,
          typename std::enable_if<
              !std::is_base_of<SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer_impl(Iterator b, Iterator e, std::random_access_iterator_tag) {
  typedef typename std::iterator_traits<Iterator>::value_type type_;
#ifdef TRISYCL_CL_LANGUAGE_VERSION
  cl::sycl::buffer<type_, 1> buf { b, e };
  buf.set_final_data(b);
#else
  size_t bufferSize = std::distance(b, e);
  // We need to copy the data back to the original Iterators when the buffer is
  // destroyed,
  // however there is temporary data midway so we have to somehow force the copy
  // back to the host.
  std::shared_ptr<type_> up{new type_[bufferSize], [b, bufferSize](type_* ptr) {
    std::copy(ptr, ptr + bufferSize, b);
    delete[] ptr;
  }};
  std::copy(b, e, up.get());
  cl::sycl::buffer<type_, 1> buf(up, cl::sycl::range<1>(bufferSize));
  buf.set_final_data(up);
#endif
  return buf;
}

/**
 *
 * @brief Creates a buffer from the given input-only iterator.
 * This buffer does not trigger a copy-back since it is just used
 * for input data.
 * @param Iterator b  Start of the range
 * @param Iterator e  End of the range
 * @param std::input_access_iterator_tag Used for iterator dispatch only
 */
template <typename Iterator,
          typename std::enable_if<
              !std::is_base_of<SyclIterator, Iterator>::value>::type* =
                                             nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer_impl(Iterator b, Iterator e, std::input_iterator_tag) {
  using type_= typename std::iterator_traits<Iterator>::value_type;
  cl::sycl::buffer<type_, 1> buf { b ,e };
  buf.set_final_data(nullptr);
  return buf;
}

/**
 *
 * @brief Extracts an existing buffer from a SYCL-enabled iterator,
 * no copy back
 * @param Iterator b  Start of the range
 * @param Iterator e  End of the range
 * @param std::input_iterator_tag Used for iterator dispatch only
 */
template <typename Iterator>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1,
                 typename Iterator::allocator_type>
reuse_buffer_impl(Iterator b, Iterator e, std::input_iterator_tag) {
  // TODO: This may need to create a sub-buffer if the range does not match
  //  the whole buffer.
  //  TODO: Technically this can be a const buffer since it is input-only
  assert((b.get_pos() == 0 && e.get_pos() == e.get_buffer().get_count()));
  return b.get_buffer();
}

/**
 *
 * @brief Extracts an existing buffer from a SYCL-enabled iterator,
 * copy back
 * @param Iterator b  Start of the range
 * @param Iterator e  End of the range
 * @param std::input_iterator_tag Used for iterator dispatch only
 */
template <typename Iterator>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1,
                 typename Iterator::allocator_type>
reuse_buffer_impl(Iterator b, Iterator e, std::random_access_iterator_tag) {
  // TODO: This may need to create a sub-buffer if the range does not match
  //  the whole buffer.
  assert((b.get_pos() == 0 && e.get_pos() == e.get_buffer().get_count()));
  return b.get_buffer();
}

/**
 *
 * @brief Calls the appropriate buffer constructor function
 *   when the iterator is a SYCL iterator.
 * @param Iterator b  Start of the range
 * @param Iterator e  End of the range
 */
template <class Iterator, typename std::enable_if<std::is_base_of<
                              SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1,
                 typename Iterator::allocator_type>
make_buffer(Iterator b, Iterator e) {
  return reuse_buffer_impl(
      b, e, typename std::iterator_traits<Iterator>::iterator_category());
}

/**
 * @brief Calls the appropriate buffer constructor function
 *   when using normal iterators
 * @param Iterator b  Start of the range
 * @param Iterator e  End of the range
 */
template <class Iterator,
          typename std::enable_if<
              !std::is_base_of<SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer(Iterator b, Iterator e) {
  return make_buffer_impl(
      b, e, typename std::iterator_traits<Iterator>::iterator_category());
}

/**
 * @brief Constructs a read-only const buffer when using SYCL iterators
 * @param Iterator b  Start of the range
 * @param Iterator e  End of the range
 */
template <class Iterator, typename std::enable_if<std::is_base_of<
                              SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1,
                 typename Iterator::allocator_type>
make_const_buffer(Iterator b, Iterator e) {
  return reuse_buffer_impl(b, e, std::input_iterator_tag());
}

/**
 * @brief Constructs a read-only const buffer when using non-sycl Iterators
 * @param Iterator b  Start of the range
 * @param Iterator e  End of the range
 */
template <class Iterator,
          typename std::enable_if<
              !std::is_base_of<SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_const_buffer(Iterator b, Iterator e) {
  return make_buffer_impl(b, e, std::input_iterator_tag());
}

/**
 * @brief Constructs a read/write sycl buffer given a type and size
 * @param size_t size
 */
template <class ElemT, int Count = 0>
cl::sycl::buffer<ElemT, 1>& make_temp_buffer(size_t size) {
  thread_local cl::sycl::buffer<ElemT, 1> buf((cl::sycl::range<1>(size)));
  if (buf.size() < size) {
      buf = cl::sycl::buffer<ElemT, 1> { cl::sycl::range<1> { 2 * size } };
  }
#ifndef TRISYCL_CL_LANGUAGE_VERSION
  buf.set_final_data(nullptr);
#endif
  return buf;
}

/**
 * @brief Constructs a read/write sycl pointer given a size and an allocation function
 */
template <typename AllocFunc, int Order = 0>
void* make_temp_pointer_impl(size_t alignment, size_t size, cl::sycl::queue& queue, AllocFunc alloc_func) {
  thread_local size_t current_size = 0;
  thread_local void* ptr = nullptr;

  if (size > current_size || ptr == nullptr) {
    void* old_ptr = ptr;
    size_t new_size = 1.5 * size + 1;
    if (alignment != 0) {
      new_size = (new_size / alignment + 1) * alignment;
    }
    void* new_ptr = alloc_func(alignment, new_size, queue);
    assert(new_ptr != nullptr);
    //queue.memcpy(new_ptr, ptr, current_size).wait();
    current_size = new_size;
    ptr = new_ptr;
    if (old_ptr != nullptr) {
        cl::sycl::free(old_ptr, queue);
    }
  }

  return ptr;
}

/**
 * @brief Constructs a read/write sycl usm pointer given a size
 */
template <int Order = 0>
void* make_temp_usm_pointer_impl(size_t alignment, size_t size, cl::sycl::queue& queue) {
  auto alloc_func = [] (auto... args) { return cl::sycl::aligned_alloc_shared(args...); };
  void* ptr = make_temp_pointer_impl<decltype(alloc_func), Order>(alignment, size, queue, alloc_func);

  // not sure if this helps
  if (ptr != nullptr && size > 0) [[likely]] {
    queue.prefetch(ptr, size); //.wait();
  }

  return ptr;
}

/**
 * @brief Constructs a read/write sycl usm pointer given a type and size
 */
template <class ElemT, int Order = 0>
ElemT* make_temp_usm_pointer(size_t size, cl::sycl::queue& queue) {
  return static_cast<ElemT*>(make_temp_usm_pointer_impl<Order>(sizeof(ElemT), sizeof(ElemT) * size, queue));
}

/**
 * @brief Constructs a read/write sycl device pointer given a size
 */
template <int Order = 0>
void* make_temp_device_pointer_impl(size_t alignment, size_t size, cl::sycl::queue& queue) {
  auto alloc_func = [] (auto... args) { return cl::sycl::aligned_alloc_device(args...); };
  return make_temp_pointer_impl<decltype(alloc_func), Order>(alignment, size, queue, alloc_func);
}

/**
 * @brief Constructs a read/write sycl device pointer given a type and size
 */
template <class ElemT, int Order = 0>
ElemT* make_temp_device_pointer(size_t size, cl::sycl::queue& queue) {
  return static_cast<ElemT*>(make_temp_device_pointer_impl<Order>(sizeof(ElemT), sizeof(ElemT) * size, queue));
}

template<typename T>
inline T read_device_pointer(const T* ptr, cl::sycl::queue& q) {
  T tmp;
  q.copy(ptr, &tmp, 1).wait();
  return tmp;
}

template<typename T>
inline void write_device_pointer(T* ptr, const T val, cl::sycl::queue& q) {
  q.copy(&val, ptr, 1).wait();
}

} /** @} namespace helpers */
} /** @} namespace sycl */

#endif  // __EXPERIMENTAL_DETAIL_SYCL_BUFFERS__

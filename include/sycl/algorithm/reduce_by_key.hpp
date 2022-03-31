#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_REDUCE_BY_KEY__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_REDUCE_BY_KEY__

#include <type_traits>
#include <algorithm>
#include <iostream>

// Detail header
#include <sycl/helpers/sycl_buffers.hpp>

#include <functional>
#include <vector>
#include <boost/iterator/zip_iterator.hpp>
#include <sycl/algorithm/exclusive_scan.hpp>
#include <sycl/algorithm/scatter_if.hpp>
#include <sycl/algorithm/transform.hpp>

#include <ZipIterator.hpp>

// reference: thrust/detail/generic/reduce_by_key.inl

namespace sycl {
namespace impl {

namespace detail
{

template <typename ValueType, typename TailFlagType, typename AssociativeOperator>
struct reduce_by_key_functor
{
  AssociativeOperator binary_op;
  
  typedef typename std::tuple<ValueType, TailFlagType> result_type;

  reduce_by_key_functor(AssociativeOperator _binary_op) : binary_op(_binary_op) {}

  result_type operator()(result_type a, result_type b) const
  {
    return result_type(std::get<1>(b) ? std::get<0>(b) : binary_op(std::get<0>(a), std::get<0>(b)),
                       std::get<1>(a) | std::get<1>(b));
  }
};


} // end namespace detail


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
std::pair<OutputIterator1, OutputIterator2>
    reduce_by_key(ExecutionPolicy &exec,
                  InputIterator1 keys_first, 
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred,
                  BinaryFunction binary_op)
{
    typedef unsigned int FlagType;  // TODO use difference_type
    using ValueType = typename std::iterator_traits<InputIterator2>::value_type;

    if (keys_first == keys_last)
        return std::make_pair(keys_output, values_output);

    cl::sycl::queue queue = exec.get_queue();

    // input size
    auto n = keys_last - keys_first;

    InputIterator2 values_last = values_first + n;
    
    // compute head flags
    FlagType* head_flags = sycl::helpers::make_temp_device_pointer<FlagType, 1>(n, queue);
    sycl::impl::transform(exec, keys_first, keys_last - 1, keys_first + 1, head_flags + 1, std::not_fn(binary_pred));
    sycl::helpers::write_device_pointer(head_flags + 0, static_cast<FlagType>(1), queue); // head_flags[0] = 1;

    // compute tail flags
    FlagType* tail_flags = sycl::helpers::make_temp_device_pointer<FlagType, 2>(n, queue); //COPY INSTEAD OF TRANSFORM
    sycl::impl::transform(exec, keys_first, keys_last - 1, keys_first + 1, tail_flags, std::not_fn(binary_pred));
    sycl::helpers::write_device_pointer(tail_flags + (n - 1), static_cast<FlagType>(1), queue); // tail_flags[n-1] = 1;

    // scan the values by flag
    ValueType* scanned_values = sycl::helpers::make_temp_device_pointer<ValueType, 0>(n, queue);
    FlagType* scanned_tail_flags = sycl::helpers::make_temp_device_pointer<FlagType, 3>(n, queue);
    
    sycl::impl::inclusive_scan
        (exec,
         ZipIter(values_first,           head_flags),
         ZipIter(values_first + n,       head_flags + n),
         ZipIter(scanned_values,         scanned_tail_flags),
         std::tuple(ValueType(), FlagType()),
         detail::reduce_by_key_functor<ValueType, FlagType, BinaryFunction>(binary_op));

    sycl::impl::exclusive_scan(exec, tail_flags, tail_flags + n, scanned_tail_flags, FlagType(0), std::plus<FlagType>());

    // number of unique keys
    FlagType N = sycl::helpers::read_device_pointer(scanned_tail_flags + (n - 1), queue) + 1;// scanned_tail_flags[n - 1] + 1;
    
    // scatter the keys and accumulated values    
    sycl::impl::scatter_if(exec, keys_first,     keys_last,          scanned_tail_flags, head_flags, keys_output);
    sycl::impl::scatter_if(exec, scanned_values, scanned_values + n, scanned_tail_flags, tail_flags, values_output);

    return std::make_pair(keys_output + N, values_output + N); 
} // end reduce_by_key()


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate>
std::pair<OutputIterator1, OutputIterator2>
    reduce_by_key(ExecutionPolicy &exec,
                  InputIterator1 keys_first, 
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred)
{
  // use plus<T> as default BinaryFunction
  typedef typename std::iterator_traits<InputIterator2>::value_type ValueType;
  return sycl::impl::reduce_by_key(exec, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, std::plus<ValueType>());
} // end reduce_by_key()


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
std::pair<OutputIterator1, OutputIterator2>
    reduce_by_key(ExecutionPolicy &exec,
                  InputIterator1 keys_first, 
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output)
{
  // use equal_to<KeyType> as default BinaryPredicate
  typedef typename std::iterator_traits<InputIterator1>::value_type KeyType;
  return sycl::impl::reduce_by_key(exec, keys_first, keys_last, values_first, keys_output, values_output, std::equal_to<KeyType>());
} // end reduce_by_key()

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_REDUCE_BY_KEY__

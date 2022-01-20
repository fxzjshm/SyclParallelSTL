#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_REDUCE_BY_KEY__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_REDUCE_BY_KEY__

#include <type_traits>
#include <algorithm>
#include <iostream>

// Detail header
#include <sycl/helpers/sycl_buffers.hpp>

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

    // input size
    auto n = keys_last - keys_first;

    InputIterator2 values_last = values_first + n;
    
    // compute head flags
    std::vector<FlagType> head_flags(n);
    sycl::impl::transform(exec, keys_first, keys_last - 1, keys_first + 1, head_flags.begin() + 1, std::not2(binary_pred));
    head_flags[0] = 1;

    // compute tail flags
    std::vector<FlagType> tail_flags(n); //COPY INSTEAD OF TRANSFORM
    sycl::impl::transform(exec, keys_first, keys_last - 1, keys_first + 1, tail_flags.begin(), std::not2(binary_pred));
    tail_flags[n-1] = 1;

    // scan the values by flag
    std::vector<ValueType> scanned_values(n);
    std::vector<FlagType> scanned_tail_flags(n);
    
    sycl::impl::inclusive_scan
        (exec,
         ZipIter(values_first,           head_flags.begin()),
         ZipIter(values_last,            head_flags.end()),
         ZipIter(scanned_values.begin(), scanned_tail_flags.begin()),
         std::tuple(ValueType(), FlagType()),
         detail::reduce_by_key_functor<ValueType, FlagType, BinaryFunction>(binary_op));

    sycl::impl::exclusive_scan(exec, tail_flags.begin(), tail_flags.end(), scanned_tail_flags.begin(), FlagType(0), std::plus());

    // number of unique keys
    FlagType N = scanned_tail_flags[n - 1] + 1;
    
    // scatter the keys and accumulated values    
    sycl::impl::scatter_if(exec, keys_first,            keys_last,             scanned_tail_flags.begin(), head_flags.begin(), keys_output);
    sycl::impl::scatter_if(exec, scanned_values.begin(), scanned_values.end(), scanned_tail_flags.begin(), tail_flags.begin(), values_output);

    return std::make_pair(keys_output + N, values_output + N); 
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
  return sycl::impl::reduce_by_key(exec, keys_first, keys_last, values_first, keys_output, values_output, std::equal_to());
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
  return sycl::impl::reduce_by_key(exec, keys_first, keys_last, values_first, keys_output, values_output, binary_pred, std::plus());
} // end reduce_by_key()

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_REDUCE_BY_KEY__

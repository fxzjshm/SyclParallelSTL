#ifndef __SYCL_IMPL_ALGORITHM_ROTATE__
#define __SYCL_IMPL_ALGORITHM_ROTATE__

#include <algorithm>
#include <vector>

#include <sycl/algorithm/copy.hpp>
#include <sycl/algorithm/rotate_copy.hpp>

namespace sycl {
namespace impl {

/* rotate.
 * Implementation of the command group that submits a rotate_copy kernel.
 */
template <class ExecutionPolicy, class ForwardIt>
ForwardIt rotate(ExecutionPolicy &sep, ForwardIt first,
                       ForwardIt middle, ForwardIt last) {
    using ValueType = typename std::iterator_traits<ForwardIt>::value_type;
    if(first == middle) return last;
    if(middle == last) return first;
    cl::sycl::queue q = sep.get_queue();
    const auto n = sycl::helpers::distance(first, last);
    if (n == 0) [[unlikely]] {
        return first;
    }
    ValueType* output_values = sycl::helpers::make_temp_device_pointer<ValueType, 0>(n, q);
    ::sycl::impl::rotate_copy(sep, first, middle, last, output_values);
    ::sycl::impl::copy(sep, output_values, output_values + n, first);
    return first + (last - middle);
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_ROTATE_COPY__

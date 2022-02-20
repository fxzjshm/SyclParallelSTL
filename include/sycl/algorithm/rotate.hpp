#ifndef __SYCL_IMPL_ALGORITHM_ROTATE__
#define __SYCL_IMPL_ALGORITHM_ROTATE__

#include <algorithm>
#include <vector>

#include <sycl/algorithm/rotate_copy.hpp>

namespace sycl {
namespace impl {

/* rotate_copy.
 * Implementation of the command group that submits a rotate_copy kernel.
 */
template <class ExecutionPolicy, class ForwardIt>
ForwardIt rotate(ExecutionPolicy &sep, ForwardIt first,
                       ForwardIt middle, ForwardIt last) {
    using ValueType = typename std::iterator_traits<ForwardIt>::value_type;
    typedef cl::sycl::usm_allocator<ValueType, cl::sycl::usm::alloc::shared> ValueTypeAllocator;
    cl::sycl::queue q = sep.get_queue();
    ValueTypeAllocator value_type_allocator(sep.get_queue());
    const auto n = std::distance(first, last);
    std::vector<ValueType, ValueTypeAllocator> output_values(n, value_type_allocator);
    ::sycl::impl::rotate_copy(sep, first, middle, last, output_values.begin());
    q.wait();
    std::copy(output_values.begin(), output_values.end(), first);
    return first + (last - middle);
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_ROTATE_COPY__

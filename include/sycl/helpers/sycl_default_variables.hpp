#ifndef __EXPERIMENTAL_DETAIL_SYCL_DEFAULT_VARIABLES__
#define __EXPERIMENTAL_DETAIL_SYCL_DEFAULT_VARIABLES__

#include <sycl/execution_policy>
#include <sycl/helpers/sycl_namegen.hpp>

namespace sycl {
namespace helpers {

struct sycl_default_variables {
  cl::sycl::device default_device;
  cl::sycl::context default_context;
  cl::sycl::queue default_queue;
  sycl::sycl_execution_policy<> default_execution_policy;
  std::mutex mutex;

  sycl_default_variables() {
    cl::sycl::device device;
    set_default_device(device);
  }

  inline void set_default_device(cl::sycl::device& device_) {
    std::lock_guard<std::mutex> lock(mutex);
    default_device = cl::sycl::device(device_);
    default_context = cl::sycl::context(device_);
    default_queue = cl::sycl::queue(default_context, default_device);
    default_execution_policy = sycl::sycl_execution_policy(default_queue);
  }
};

inline sycl_default_variables& get_sycl_default_variables() {
  static sycl_default_variables vars;
  return vars;
}

inline void set_default_device(cl::sycl::device& device_) {
  get_sycl_default_variables().set_default_device(device_);
}

inline cl::sycl::device& default_device() {
  return get_sycl_default_variables().default_device;
}

inline cl::sycl::context& default_context() {
  return get_sycl_default_variables().default_context;
}

inline cl::sycl::queue& default_queue() {
  return get_sycl_default_variables().default_queue;
}

inline sycl::sycl_execution_policy<>& default_execution_policy() {
  return get_sycl_default_variables().default_execution_policy;
}

}  // namespace helpers
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_SYCL_DEFAULT_VARIABLES__
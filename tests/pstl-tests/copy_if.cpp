#include "gmock/gmock.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include <experimental/algorithm>
#include <sycl/execution_policy>

#include <sycl/helpers/sycl_usm_vector.hpp>

// Todo: move to execution_policy
#include <sycl/algorithm/copy_if.hpp>

namespace parallel = std::experimental::parallel;

struct CopyIfAlgorithm : public testing::Test {};

TEST_F(CopyIfAlgorithm, CopyIf) {
  sycl::helpers::usm_vector<int> input = {2, 4, 6, 8, 10, 12, 14, 16}, output1(input.size()), output2(input.size());
  auto predicate = [] (auto x) { return x % 4 != 0; };

  std::copy_if(input.begin(), input.end(), output1.begin(), predicate);

  sycl::sycl_execution_policy<class CopyIf> snp;
  sycl::impl::copy_if(snp, input.begin(), input.end(), output2.begin(), predicate);

  EXPECT_TRUE(std::equal(output1.begin(), output1.end(), output2.begin()));
}

TEST_F(CopyIfAlgorithm, CopyIfLong) {
  const size_t size = 1<<20;
  sycl::helpers::usm_vector<int> input(size), output1(size), output2(size);
  auto predicate = [] (auto x) { return x % 3 == 0; };
  std::generate(input.begin(), input.end(), std::rand);

  std::copy_if(input.begin(), input.end(), output1.begin(), predicate);

  sycl::sycl_execution_policy<class CopyIfLong> snp;
  sycl::impl::copy_if(snp, input.begin(), input.end(), output2.begin(), predicate);

  EXPECT_TRUE(std::equal(output1.begin(), output1.end(), output2.begin()));
}

template <class InputIterator, class StencilIterator, class OutputIterator, class UnaryPredicate>
  OutputIterator __copy_if (InputIterator first, InputIterator last, StencilIterator stencil,
                          OutputIterator result, UnaryPredicate pred)
{
  while (first!=last) {
    if (pred(*stencil)) {
      *result = *first;
      ++result;
    }
    ++first;
    ++stencil;
  }
  return result;
}

TEST_F(CopyIfAlgorithm, CopyIfLongWithStencil) {
  const size_t size = 1<<20;
  sycl::helpers::usm_vector<int> input(size), output1(size), output2(size);
  sycl::helpers::usm_vector<float> stencil(size);
  auto predicate = [] (auto x) { return x < 0.5f; };

  std::generate(input.begin(), input.end(), std::rand);
  std::generate(stencil.begin(), stencil.end(), 
                [](){ return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX); });

  ::__copy_if(input.begin(), input.end(), stencil.begin(), output1.begin(), predicate);

  sycl::sycl_execution_policy<class CopyIfLongWithStencil> snp;
  sycl::impl::copy_if(snp, input.begin(), input.end(), stencil.begin(), output2.begin(), predicate);

  EXPECT_TRUE(std::equal(output1.begin(), output1.end(), output2.begin()));
}

/*
TEST_F(CopyIfAlgorithm, CopyIfLongAllTrue) {
  const size_t size = 1<<20;
  sycl::helpers::usm_vector<int> input(size), output(size);
  auto predicate = [] (auto x) { return true; };

  std::generate(input.begin(), input.end(), std::rand);

  sycl::sycl_execution_policy<class CopyIfLongAllTrue> snp;

  int n = 100;
  while(n--) {
    sycl::impl::copy_if(snp, input.begin(), input.end(), output.begin(), predicate);
    EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin()));
  }
}
*/

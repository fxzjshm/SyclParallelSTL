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
#include "gmock/gmock.h"

#include <vector>
#include <algorithm>
#include <iostream>

#include <sycl/execution_policy>
#include <experimental/algorithm>

#include <sycl/helpers/sycl_usm_vector.hpp>

using namespace std::experimental::parallel;

struct SortAlgorithm : public testing::Test {
  sycl::sycl_execution_policy<>* sycl_policy;

  void SetUp() { sycl_policy = new sycl::sycl_execution_policy<>(); }

  void TearDown() { delete sycl_policy; }
};

TEST_F(SortAlgorithm, TestStdSort) {
  sycl::helpers::usm_vector<int> v = {2, 1, 3};

  std::sort(v.begin(), v.end());

  EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST_F(SortAlgorithm, TestSyclSort) {
  {
    sycl::helpers::usm_vector<int> v(64);
    std::generate(v.begin(), v.end(),
                  std::rand);  // Using the C function rand()
    // The bitonic sort is triggered with a power of two vector size
    sort(*sycl_policy, v.begin(), v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
  }

  {
    sycl::helpers::usm_vector<int> v(3);
    std::generate(v.begin(), v.end(),
                  std::rand);  // Using the C function rand()
    // The bitonic sort is triggered with a power of two vector size
    sort(*sycl_policy, v.begin(), v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
  }

  {
    sycl::helpers::usm_vector<int> a(10);
    std::generate(a.begin(), a.end(), std::rand);
    sort(*sycl_policy, a.begin(), a.end());
    EXPECT_TRUE(std::is_sorted(a.begin(), a.end()));
  }

  {
    sycl::helpers::usm_vector<int> v(2333);
    std::generate(v.begin(), v.end(), std::rand);
    sycl::helpers::usm_vector<int> v2(v.begin(), v.end());
    sort(*sycl_policy, v.begin(), v.end());
    std::sort(v2.begin(), v2.end());
    EXPECT_TRUE(std::equal(v.begin(), v.end(), v2.begin()));
  }

  {
    sycl::helpers::usm_vector<int> v(233);
    std::generate(v.begin(), v.end(), std::rand);
    sycl::helpers::usm_vector<int> v2(v.begin(), v.end());
    sort(*sycl_policy, v.begin(), v.end());
    std::sort(v2.begin(), v2.end());
    EXPECT_TRUE(std::equal(v.begin(), v.end(), v2.begin()));
  }
}

TEST_F(SortAlgorithm, TestSycl2Sort) {
  sycl::helpers::usm_vector<int> v = {2, 1, 3, 7, 9, 5, 4};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SortAlgorithm2> snp(q);
  auto cmp = [=](int a, int b) { return a >= b; };
  sort(snp, v.begin(), v.end(), cmp);

  EXPECT_TRUE(std::is_sorted(v.begin(), v.end(), cmp));
}

TEST_F(SortAlgorithm, TestSycl3Sort) {
  sycl::helpers::usm_vector<int> v = {2, 1, 3, 7, 9, 5, 4, 6};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class SortAlgorithm3> snp(q);
  auto cmp = [=](int a, int b) { return a >= b; };
  sort(snp, v.begin(), v.end(), cmp);

  EXPECT_TRUE(std::is_sorted(v.begin(), v.end(), cmp));
}

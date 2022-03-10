#include "gmock/gmock.h"

#include <vector>
#include <algorithm>
#include <numeric>

#include <sycl/execution_policy>
#include <experimental/algorithm>

// TODO move to execution_policy after test
#include <sycl/algorithm/reduce_by_key.hpp>

#include <sycl/helpers/sycl_usm_vector.hpp>

using namespace std::experimental::parallel;

// adapted from Boost.Compute

class ReduceByKeyAlgorithm : public testing::Test {
 public:
};

sycl::sycl_execution_policy exec;

template<typename T>
struct min_functor {
    T operator() (const T x, const T y) const {
        return std::min(x, y);
    }
};

template<typename T>
struct max_functor {
    T operator() (const T x, const T y) const {
        return std::max(x, y);
    }
};

TEST_F(ReduceByKeyAlgorithm, reduce_by_key_int)
{
    //! [reduce_by_key_int]
    // setup keys and values
    int keys[] = { 0, 2, -3, -3, -3, -3, -3, 4 };
    int data[] = { 1, 1, 1, 1, 1, 2, 5, 1 };
    
    sycl::helpers::usm_vector<int> keys_input(keys, keys + 8);
    sycl::helpers::usm_vector<int> values_input(data, data + 8);
    
    sycl::helpers::usm_vector<int> keys_output(8);
    sycl::helpers::usm_vector<int> values_output(8);

    // reduce by key
    size_t count = sycl::impl::reduce_by_key(exec,
                                             keys_input.begin(), keys_input.end(), values_input.begin(),
                                             keys_output.begin(), values_output.begin()).first - keys_output.begin();
    // keys_output = { 0, 2, -3, 4 }
    // values_output = { 1, 1, 10, 1 }
    sycl::helpers::usm_vector<int> keys_output_expected = { 0, 2, -3, 4 };
    sycl::helpers::usm_vector<int> values_output_expected = { 1, 1, 10, 1 };
    //! [reduce_by_key_int]
    EXPECT_TRUE(std::equal(begin(keys_output), begin(keys_output) + count, begin(keys_output_expected)));
    EXPECT_TRUE(std::equal(begin(values_output), begin(values_output) + count, begin(values_output_expected)));
}

TEST_F(ReduceByKeyAlgorithm, reduce_by_key_int_long_vector)
{
    size_t size = 1024;
    sycl::helpers::usm_vector<int> keys_input(size, int(0));
    sycl::helpers::usm_vector<int> values_input(size, int(1));

    sycl::helpers::usm_vector<int> keys_output(size);
    sycl::helpers::usm_vector<int> values_output(size);

    sycl::impl::reduce_by_key(exec,
                              keys_input.begin(), keys_input.end(), values_input.begin(),
                              keys_output.begin(), values_output.begin());

    EXPECT_TRUE(keys_output[0] == 0);
    EXPECT_TRUE(values_output[0] == (static_cast<int>(size)));

    keys_input[137] = 1;
    keys_input[677] = 1;
    keys_input[1001] = 1;
    inclusive_scan(exec, keys_input.begin(), keys_input.end(), keys_input.begin());

    size_t count = sycl::impl::reduce_by_key(exec,
                                             keys_input.begin(), keys_input.end(), values_input.begin(),
                                             keys_output.begin(), values_output.begin()).first - keys_output.begin();

    sycl::helpers::usm_vector<int> keys_output_expected = { 0, 1, 2, 3 };
    sycl::helpers::usm_vector<int> values_output_expected = { 137, 540, 324, 23 };

    EXPECT_TRUE(std::equal(begin(keys_output), begin(keys_output) + count, begin(keys_output_expected)));
    EXPECT_TRUE(std::equal(begin(values_output), begin(values_output) + count, begin(values_output_expected)));
}

TEST_F(ReduceByKeyAlgorithm, reduce_by_key_empty_vector)
{
    sycl::helpers::usm_vector<int> keys_input;
    sycl::helpers::usm_vector<int> values_input;

    sycl::helpers::usm_vector<int> keys_output;
    sycl::helpers::usm_vector<int> values_output;

    sycl::impl::reduce_by_key(exec,
                      keys_input.begin(), keys_input.end(), values_input.begin(),
                      keys_output.begin(), values_output.begin());

    EXPECT_TRUE(keys_output.empty());
    EXPECT_TRUE(values_output.empty());
}

TEST_F(ReduceByKeyAlgorithm, reduce_by_key_int_one_key_value)
{
    int keys[] = { 22 };
    int data[] = { -9 };

    sycl::helpers::usm_vector<int> keys_input(keys, keys + 1);
    sycl::helpers::usm_vector<int> values_input(data, data + 1);

    sycl::helpers::usm_vector<int> keys_output(1);
    sycl::helpers::usm_vector<int> values_output(1);

    sycl::impl::reduce_by_key(exec,
                      keys_input.begin(), keys_input.end(), values_input.begin(),
                      keys_output.begin(), values_output.begin());

    EXPECT_TRUE(keys_output[0] == 22);
    EXPECT_TRUE(values_output[0] == -9);
}

TEST_F(ReduceByKeyAlgorithm, reduce_by_key_int_min_max)
{
    int keys[] = { 0, 2, 2, 3, 3, 3, 3, 3, 4 };
    int data[] = { 1, 2, 1, -3, 1, 4, 2, 5, 77 };

    sycl::helpers::usm_vector<int> keys_input(keys, keys + 9);
    sycl::helpers::usm_vector<int> values_input(data, data + 9);

    sycl::helpers::usm_vector<int> keys_output(9);
    sycl::helpers::usm_vector<int> values_output(9);

    size_t count = sycl::impl::reduce_by_key(exec,
                                             keys_input.begin(), keys_input.end(), values_input.begin(),
                                             keys_output.begin(), values_output.begin(), std::equal_to<int>(), min_functor<int>()
                                            ).first - keys_output.begin();

    sycl::helpers::usm_vector<int> keys_output_expected = { 0, 2, 3,  4 };
    sycl::helpers::usm_vector<int> values_output_expected = { 1, 1, -3, 77 };

    EXPECT_TRUE(std::equal(begin(keys_output), begin(keys_output) + count, begin(keys_output_expected)));
    EXPECT_TRUE(std::equal(begin(values_output), begin(values_output) + count, begin(values_output_expected)));

    count = sycl::impl::reduce_by_key(exec,
                                      keys_input.begin(), keys_input.end(), values_input.begin(),
                                      keys_output.begin(), values_output.begin(), std::equal_to<int>(), max_functor<int>()
                                     ).first - keys_output.begin();

    keys_output_expected = { 0, 2, 3,  4 };
    values_output_expected = { 1, 2, 5, 77 };

    EXPECT_TRUE(std::equal(begin(keys_output), begin(keys_output) + count, begin(keys_output_expected)));
    EXPECT_TRUE(std::equal(begin(values_output), begin(values_output) + count, begin(values_output_expected)));
}

TEST_F(ReduceByKeyAlgorithm, reduce_by_key_float_max)
{
    int keys[] = { 0, 2, 2, 3, 3, 3, 3, 3, 4 };
    float data[] = { 1.0, 2.0, -1.5, -3.0, 1.0, -0.24, 2, 5, 77.1 };

    sycl::helpers::usm_vector<int> keys_input(keys, keys + 9);
    sycl::helpers::usm_vector<float> values_input(data, data + 9);

    sycl::helpers::usm_vector<int> keys_output(9);
    sycl::helpers::usm_vector<float> values_output(9);

    size_t count = sycl::impl::reduce_by_key(exec,
                                             keys_input.begin(), keys_input.end(), values_input.begin(),
                                             keys_output.begin(), values_output.begin(), std::equal_to<int>(), max_functor<int>()
                                            ).first - keys_output.begin();

    sycl::helpers::usm_vector<int> keys_output_expected = { 0, 2, 3,  4 };
    EXPECT_TRUE(std::equal(begin(keys_output), begin(keys_output) + count, begin(keys_output_expected)));
    EXPECT_TRUE(std::abs(float(values_output[0]) - 1.0f) < 1e-4f);
    EXPECT_TRUE(std::abs(float(values_output[1]) - 2.0f) < 1e-4f);
    EXPECT_TRUE(std::abs(float(values_output[2]) - 5.0f) < 1e-4f);
    EXPECT_TRUE(std::abs(float(values_output[3]) - 77.1f) < 1e-4f);
}

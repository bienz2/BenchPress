// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.

#include "gtest/gtest.h"
#include "ping_pong/ping_pong_profiler.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int temp=RUN_ALL_TESTS();
    MPI_Finalize();
    return temp;
} // end of main() //

TEST(CollectiveTest, TestsInCollective)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    profile_ping_pong(2, 10);

#ifdef CUDA_AWARE
    profile_ping_pong_gpu(2, 10);
#endif

    profile_max_rate(false, 2, 10);

#ifdef CUDA_AWARE
    profile_max_rate_gpu(false, 2, 10);
#endif

    profile_ping_pong_mult(2, 10);

#ifdef CUDA_AWARE
    profile_ping_pong_mult_gpu(2, 10);
#endif

} // end of  TEST(ParStrengthTest, TestsInTests) //

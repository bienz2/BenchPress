// EXPECT_EQ and ASSERT_EQ are macros
// EXPECT_EQ test execution and continues even if there is a failure
// ASSERT_EQ test execution and aborts if there is a failure
// The ASSERT_* variants abort the program execution if an assertion fails 
// while EXPECT_* variants continue with the run.

#include "gtest/gtest.h"
#include "collective/alltoallv_profiler.h"

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

#ifdef CUDA_AWARE
    alltoallv_profile_cuda_aware(2);
#endif

    alltoallv_profile_3step(2);
    alltoallv_profile_3step_extra_msg(2);
    alltoallv_profile_3step_dup_devptr(2);


} // end of  TEST(ParStrengthTest, TestsInTests) //

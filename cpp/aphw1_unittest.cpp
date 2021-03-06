#include <limits.h>
#include "aphw1.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include "gtest/gtest.h"
namespace
{
TEST(APHW1Test, getDataFunctionTest)
{
    std::vector<std::vector<double>> data{getData("AP-Data.csv", 1)};
    EXPECT_EQ(234, data.size());
    EXPECT_EQ(1, data[0][0]);
    EXPECT_EQ(1, data[0][7]);
    EXPECT_EQ(22, data[25][4]);
    EXPECT_EQ(0.89, data[209][6]);
    EXPECT_EQ(0.23, data[233][2]);
    EXPECT_EQ(0, data[233][7]);
}

TEST(APHW1Test, displayFunctionTest)
{
    std::vector<std::vector<double>> data {getData("AP-Data.csv", 1)};
    displayDataset(data, true);
}

TEST(APHW1Test, hFunctionTest)
{
    std::vector<double> student{1, 2, 3, 4, 5, 6, 7};
    std::vector<double> w (7, 0);
    EXPECT_EQ(0.5, h(student, w));

    // std::vector<double> student1{1 , 1 , 2, 3, 4, 5, 6, 7};
    // std::vector<double> w1{0.1,0.2,0.3,0.4,0.5,0.6,0.7};
    // EXPECT_EQ(0.999988, h(student1, w1));    
}

TEST(APHW1Test, costFunctionTest)
{
    std::vector<std::vector<double>> data{getData("AP-Data.csv", 1)};
    std::vector<double> w (7, 0);
    std::vector<size_t> indices {0};
    double j{J(data, indices, w)};
    EXPECT_TRUE(std::abs(j - 0.693) < 0.001);
}

TEST(APHW1Test, fitOneEpochFunctionTest)
{
    std::vector<std::vector<double>> data{getData("AP-Data.csv",1)};
    std::vector<double> w (7, 0);
    w = fitOneEpoch(data, w);
    EXPECT_TRUE(std::abs(w[0] - -0.0236) < 0.01);
}

TEST(APHW1Test, fitFunctionTest)
{
    std::vector<std::vector<double>> data{getData("AP-Data.csv", 1)};
    std::vector<double> w (7, 0);
    w = fit(data, w, 0.01, 3000, 8, true);
    EXPECT_TRUE(std::abs(w[0] - -16.9) < 0.1);
}

TEST(APHW1Test, predictFunctionTest)
{
    std::vector<std::vector<double>> data{getData("AP-Data.csv", 1)};
    std::vector<double> w (7, 0);
    w = fit(data, w, 0.01, 3000, 8, false);
    std::vector<double> outputs{predict(data, w, true)};
    EXPECT_TRUE(std::abs(outputs[0] - 0.848) < 0.01);
    EXPECT_TRUE(std::abs(outputs[1] - 0.997) < 0.01);
}
}

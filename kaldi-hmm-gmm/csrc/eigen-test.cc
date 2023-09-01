// kaldi-hmm-gmm/csrc/eigen-test.cc

// Copyright (c)  2023     Xiaomi Corporation

#include "Eigen/Dense"
#include "gtest/gtest.h"

TEST(Eigen, Hello) {
  Eigen::MatrixXd m(2, 2);  // uninitialized; contains garbage data
  EXPECT_EQ(m.size(), 2 * 2);

  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);

  auto m2 = m;  // value semantics; create a copy
  m2(0, 0) = 10;
  EXPECT_EQ(m(0, 0), 3);

  double *d = &m(0, 0);
  d[0] = 11;
  d[1] = 20;
  d[2] = 30;
  d[3] = 40;
  EXPECT_EQ(m(0, 0), 11);  // column major by default
  EXPECT_EQ(m(1, 0), 20);
  EXPECT_EQ(m(0, 1), 30);  // it is contiguous in memory
  EXPECT_EQ(m(1, 1), 40);

  // column major
  EXPECT_EQ(m(0), 11);
  EXPECT_EQ(m(1), 20);
  EXPECT_EQ(m(2), 30);
  EXPECT_EQ(m(3), 40);

  Eigen::MatrixXf a;
  EXPECT_EQ(a.size(), 0);

  Eigen::Matrix3f b;  // uninitialized
  EXPECT_EQ(b.size(), 3 * 3);

  Eigen::MatrixXf c(2, 5);  // uninitialized
  EXPECT_EQ(c.size(), 2 * 5);

  {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> f{
        {1, 2},
        {3, 4},
    };

    // row major
    EXPECT_EQ(f(0), 1);
    EXPECT_EQ(f(1), 2);
    EXPECT_EQ(f(2), 3);
    EXPECT_EQ(f(3), 4);

    // Note: f[0] causes compilation errors
  }
}

TEST(Eigen, Identity) {
  auto m = Eigen::Matrix3f::Identity();  // 3x3 identity matrix
  EXPECT_EQ(m.sum(), 3);

  auto n = Eigen::MatrixXf::Identity(2, 3);  // 2x3 identity matrix
#if 0
  1 0 0
  0 1 0
#endif
}

// https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae814abb451b48ed872819192dc188c19
TEST(Eigen, Random) {
  // Random: Uniform distribution in the range [-1, 1]
  auto m = Eigen::MatrixXd::Random(2, 3);
#if 0
 -0.999984   0.511211  0.0655345
 -0.736924 -0.0826997  -0.562082
#endif

  // Note: We don't need to specify the shape for Random() in this case
  auto m2 = Eigen::Matrix3d::Random();
#if 0
 -0.999984 -0.0826997  -0.905911
 -0.736924  0.0655345   0.357729
  0.511211  -0.562082   0.358593
#endif
}

TEST(Eigen, Vector) {
  Eigen::VectorXd v(3);

  // comma initializer
  v << 1, 2, 3;
  EXPECT_EQ(v(0), 1);
  EXPECT_EQ(v(1), 2);
  EXPECT_EQ(v(2), 3);

  // vector also support operator[]
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 2);
  EXPECT_EQ(v[2], 3);

  double *p = &v[0];
  p[0] = 10;
  p[1] = 20;
  p[2] = 30;

  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);

  // fixed size
  Eigen::Vector3d a(10, 20, 30);
  EXPECT_EQ(a[0], 10);
  EXPECT_EQ(a[1], 20);
  EXPECT_EQ(a[2], 30);
}

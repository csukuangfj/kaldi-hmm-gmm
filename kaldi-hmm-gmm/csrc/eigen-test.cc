// kaldi-hmm-gmm/csrc/eigen-test.cc

// Copyright (c)  2023     Xiaomi Corporation

#include <type_traits>

#include "Eigen/Dense"
#include "gtest/gtest.h"

// See
//
// Quick reference guide
// https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
//
// Preprocessor directives
// https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
//
// Understanding Eigen
// https://eigen.tuxfamily.org/dox/UserManual_UnderstandingEigen.html
//
// Using Eigen in CMake Projects
// https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html

TEST(Eigen, Hello) {
  Eigen::MatrixXd m(2, 2);  // uninitialized; contains garbage data
  EXPECT_EQ(m.size(), 2 * 2);
  EXPECT_EQ(m.rows(), 2);
  EXPECT_EQ(m.cols(), 2);

  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);

  auto m2 = m;  // value semantics; create a copy
  m2(0, 0) = 10;
  EXPECT_EQ(m(0, 0), 3);

  Eigen::MatrixXd m3 = std::move(m2);
  // now m2 is empty
  EXPECT_EQ(m2.size(), 0);
  EXPECT_EQ(m3(0, 0), 10);

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

  EXPECT_EQ(c.rows(), 2);
  EXPECT_EQ(c.cols(), 5);

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

TEST(Eigen, CommaInitializer) {
  // comma initializer does not depend on the storage major
  {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m(2,
                                                                            2);
    m << 1, 2, 3, 4;
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(1, 0), 3);
    EXPECT_EQ(m(1, 1), 4);
  }

  {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(2,
                                                                            2);
    m << 1, 2, 3, 4;
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(1, 0), 3);
    EXPECT_EQ(m(1, 1), 4);
  }
}

TEST(Eigen, Resize) {
  // a resize operation is a destructive operation if it changes the size.
  // The original content is not copied to the resized area
  Eigen::MatrixXf a(2, 3);
  EXPECT_EQ(a.rows(), 2);
  EXPECT_EQ(a.cols(), 3);
  EXPECT_EQ(a.size(), a.rows() * a.cols());

  a.resize(5, 6);
  EXPECT_EQ(a.rows(), 5);
  EXPECT_EQ(a.cols(), 6);
  EXPECT_EQ(a.size(), a.rows() * a.cols());

  Eigen::MatrixXf b;
  EXPECT_EQ(b.size(), 0);

  b = a;  // copy by value
  EXPECT_EQ(b.rows(), 5);
  EXPECT_EQ(b.cols(), 6);
}

TEST(Eigen, MatMul) {
  Eigen::MatrixXf a(2, 2);
  a << 1, 2, 3, 4;

  Eigen::MatrixXf b(2, 2);
  b << 3, 0, 0, 2;

  Eigen::MatrixXf c = a * b;  // matrix multiplication
  EXPECT_EQ(c(0, 0), a(0, 0) * b(0, 0));
  EXPECT_EQ(c(0, 1), a(0, 1) * b(1, 1));

  EXPECT_EQ(c(1, 0), a(1, 0) * b(0, 0));
  EXPECT_EQ(c(1, 1), a(1, 1) * b(1, 1));

  Eigen::MatrixXf d;
  d.noalias() = a * b;  // explicitly specify that there is no alias

  EXPECT_EQ(d(0, 0), a(0, 0) * b(0, 0));
  EXPECT_EQ(d(0, 1), a(0, 1) * b(1, 1));

  EXPECT_EQ(d(1, 0), a(1, 0) * b(0, 0));
  EXPECT_EQ(d(1, 1), a(1, 1) * b(1, 1));
}

TEST(Eigen, Transpose) {
  Eigen::MatrixXf a(2, 2);
  a << 1, 2, 3, 4;

  a = a.transpose();  // wrong due to alias
#if 0
  1 2
  2 4
#endif

  Eigen::MatrixXf b(2, 2);
  b << 1, 2, 3, 4;
  b.transposeInPlace();  // correct
#if 0
  1 3
  2 4
#endif
}

TEST(Eigen, Reduction) {
  Eigen::MatrixXf m(2, 2);
  m << 1, 2, 3, -5;
#if 0
  1 2
  3 -5
#endif

  EXPECT_EQ(m.sum(), 1);
  EXPECT_EQ(m.prod(), -30);
  EXPECT_EQ(m.mean(), m.sum() / m.size());
  EXPECT_EQ(m.minCoeff(), -5);
  EXPECT_EQ(m.maxCoeff(), 3);
  EXPECT_EQ(m.trace(), 1 + (-5));
  EXPECT_EQ(m.trace(), m.diagonal().sum());

  std::ptrdiff_t row_id, col_id;

  float a = m.minCoeff(&row_id, &col_id);
  EXPECT_EQ(a, -5);
  EXPECT_EQ(row_id, 1);
  EXPECT_EQ(col_id, 1);

  float b = m.maxCoeff(&row_id, &col_id);
  EXPECT_EQ(b, 3);
  EXPECT_EQ(row_id, 1);
  EXPECT_EQ(col_id, 0);
}

TEST(Eigen, Array) {
  // Note: It is XX for a 2-D array
  Eigen::ArrayXXf a(2, 3);
  a << 1, 2, 3, 4, 5, 6;
#if 0
  1 2 3
  4 5 6
#endif

  EXPECT_EQ(a(0, 0), 1);
  EXPECT_EQ(a(0, 1), 2);
  EXPECT_EQ(a(0, 2), 3);

  EXPECT_EQ(a(1, 0), 4);
  EXPECT_EQ(a(1, 1), 5);
  EXPECT_EQ(a(1, 2), 6);

  EXPECT_EQ(a.rows(), 2);
  EXPECT_EQ(a.cols(), 3);

  Eigen::Array<float, 5, 2> b;
  EXPECT_EQ(b.rows(), 5);
  EXPECT_EQ(b.cols(), 2);

  // 1-d array
  Eigen::ArrayXf c(10);

  EXPECT_EQ(c.rows(), 10);
  EXPECT_EQ(c.cols(), 1);
  EXPECT_EQ(c.size(), 10);
  static_assert(
      std::is_same<decltype(c), Eigen::Array<float, Eigen::Dynamic, 1>>::value,
      "");

  static_assert(std::is_same<Eigen::ArrayXf,
                             Eigen::Array<float, Eigen::Dynamic, 1>>::value,
                "");

  static_assert(
      std::is_same<Eigen::ArrayXXf,
                   Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>::value,
      "");

  static_assert(std::is_same<Eigen::Array3f, Eigen::Array<float, 3, 1>>::value,
                "");

  static_assert(std::is_same<Eigen::Array33f, Eigen::Array<float, 3, 3>>::value,
                "");
}

TEST(Eigen, ArrayMultiplication) {
  Eigen::ArrayXXf a(2, 2);
  a << 1, 2, 3, 4;
  Eigen::ArrayXXf b = a * a;

  EXPECT_EQ(b(0, 0), a(0, 0) * a(0, 0));
  EXPECT_EQ(b(0, 1), a(0, 1) * a(0, 1));
  EXPECT_EQ(b(1, 0), a(1, 0) * a(1, 0));
  EXPECT_EQ(b(1, 1), a(1, 1) * a(1, 1));

  // column-wise product
  Eigen::ArrayXXf c = a.matrix().cwiseProduct(a.matrix());

  EXPECT_EQ(c(0, 0), a(0, 0) * a(0, 0));
  EXPECT_EQ(c(0, 1), a(0, 1) * a(0, 1));
  EXPECT_EQ(c(1, 0), a(1, 0) * a(1, 0));
  EXPECT_EQ(c(1, 1), a(1, 1) * a(1, 1));
}

TEST(Eigen, CoefficientWise) {
  Eigen::ArrayXXf a(2, 2);
  a << 1, 2, 3, -4;

  EXPECT_EQ(a.abs()(1, 1), 4);
  EXPECT_EQ(a.abs().sum(), 10);

  EXPECT_EQ(a.abs().sqrt()(1, 1), 2);
}

TEST(Eigen, Row) {
  Eigen::MatrixXf m(2, 3);
  m << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXf a = m.row(0);  // copied to a
  EXPECT_EQ(a.rows(), 1);
  EXPECT_EQ(a.cols(), 3);

  a(0) = 10;
  EXPECT_EQ(m(0, 0), 1);

  Eigen::MatrixXf b = m.col(1);  // copied to b
  EXPECT_EQ(b.rows(), 2);
  EXPECT_EQ(b.cols(), 1);
  b(0) = 10;
  EXPECT_EQ(m(0, 1), 2);

  auto c = m.row(0);  // c is a proxy object; no copy is created
  c(0) = 10;          // also change m
  EXPECT_EQ(m(0, 0), 10);
  EXPECT_EQ(c.rows(), 1);
  EXPECT_EQ(c.cols(), 3);

  auto d = c;  // d is also a proxy
  d(0) = 100;
  EXPECT_EQ(c(0), 100);
  EXPECT_EQ(m(0), 100);

  // N5Eigen6MatrixIfLin1ELin1ELi0ELin1ELin1EEE
  // std::cout << typeid(m).name() << "\n";

  // N5Eigen6MatrixIfLin1ELin1ELi0ELin1ELin1EEE
  // std::cout << typeid(b).name() << "\n";

  // N5Eigen5BlockINS_6MatrixIfLin1ELin1ELi0ELin1ELin1EEELi1ELin1ELb0EEE
  // std::cout << typeid(c).name() << "\n";
}

TEST(Eigen, SpecialFunctions) {
  Eigen::MatrixXf a(2, 3);
  a.setOnes();
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a(i), 1);
  }

  a.setZero();
  for (int32_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a(i), 0);
  }
}

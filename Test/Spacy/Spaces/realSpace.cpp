// Copyright (C) 2015 by Lars Lubkoll. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#include <gtest/gtest.h>

#include "Spacy/vector.hh"
#include "Spacy/Spaces/realSpace.hh"
#include "Spacy/Util/cast.hh"

TEST(RealSpaceTest, DefaultIndex)
{
  auto r = Spacy::Real{2.};
  ASSERT_EQ( r.space().index() , 0 );
}

TEST(RealSpaceTest,ElementTest)
{
  using namespace Spacy;
  auto R = RealSpace::makeHilbertSpace();
  auto x = zero(R);
  EXPECT_DOUBLE_EQ( toDouble(x) , 0. );
}

TEST(RealSpaceTest,ScalarProductTest)
{
  using namespace Spacy;
  auto R = RealSpace::makeHilbertSpace();
  auto x = zero(R);
  auto y = zero(R);
  toDouble(x) = 1;
  toDouble(y) = -2;
  EXPECT_DOUBLE_EQ( toDouble(x), 1. );
  EXPECT_DOUBLE_EQ( toDouble(y), -2. );
  EXPECT_DOUBLE_EQ( toDouble(x*y), -2. );
  EXPECT_DOUBLE_EQ( toDouble(x*y), toDouble(R.scalarProduct()(x,y)) );
  EXPECT_TRUE( R.isHilbertSpace() );
}

TEST(RealSpaceTest,NormTest)
{
  using namespace Spacy;
  auto R = RealSpace::makeHilbertSpace();
  auto x = zero(R);
  auto y = zero(R);
  toDouble(x) = 1;
  toDouble(y) = -2;
  EXPECT_DOUBLE_EQ( toDouble(x), 1. );
  EXPECT_DOUBLE_EQ( toDouble(y), -2. );
  EXPECT_DOUBLE_EQ( toDouble(R.norm()(x)) , 1. );
  EXPECT_DOUBLE_EQ( toDouble(R.norm()(y)) , 2. );
}

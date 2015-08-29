#include "vectorSpace.hh"

#include "Algorithms/VectorSpaces/RealSpace/vector.hh"
#include "Algorithms/VectorSpaces/RealSpace/scalarProduct.hh"

::Algorithm::VectorSpace Algorithm::Real::makeHilbertSpace()
{
  return ::Algorithm::makeHilbertSpace( [](const VectorSpace* space){ return Vector{*space}; } ,
                                        ScalarProduct{} );
}
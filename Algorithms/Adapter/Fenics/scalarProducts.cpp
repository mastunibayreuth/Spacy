#include "scalarProducts.hh"

#include "vector.hh"
#include "Util/castTo.hh"

#include <memory>

namespace Algorithm
{
  using Interface::AbstractFunctionSpaceElement;

  namespace Fenics
  {
    double l2ScalarProduct::operator()(const AbstractFunctionSpaceElement& x, const AbstractFunctionSpaceElement& y) const
    {
      return castTo<Vector>(x).impl().vector()->inner( *castTo<Vector>(y).impl().vector() );
    }

    ScalarProduct::ScalarProduct(std::shared_ptr<dolfin::GenericMatrix> A)
      : A_(A)
    {}

    double ScalarProduct::operator()(const AbstractFunctionSpaceElement& x, const AbstractFunctionSpaceElement& y) const
    {
      auto x_ = std::make_shared<dolfin::Vector>(A_->mpi_comm(), A_->size(0));
      copy(x,*x_);
      auto Ax = x_->copy();
      A_->mult(*x_, *Ax);

      return Ax->inner( *x_ );
    }

  }
}

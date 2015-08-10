#include "trackingTypeCostFunctional.hh"

#include "functionSpace.hh"

#include "Interface/abstractFunctionSpace.hh"

#include "FunctionSpaces/RealNumbers/real.hh"
#include "FunctionSpaces/ProductSpace/productSpaceElement.hh"
#include "Util/Exceptions/invalidArgumentException.hh"
#include "Util/castTo.hh"

#include <utility>

namespace Algorithm
{
  using Interface::AbstractFunctionSpace;
  using Interface::AbstractFunctionSpaceElement;

  TrackingTypeCostFunctional::TrackingTypeCostFunctional(double alpha, const AbstractFunctionSpaceElement &referenceState,
                                                         std::shared_ptr<AbstractFunctionSpace> domain,
                                                         std::unique_ptr<Interface::AbstractOperator>&& My,
                                                         std::unique_ptr<Interface::AbstractOperator>&& Mu)
    : AbstractFunctional(domain),
      alpha_(alpha),
      referenceState_( clone(referenceState) ),
      My_(std::move(My)),
      Mu_(std::move(Mu))
  {}

  TrackingTypeCostFunctional::TrackingTypeCostFunctional(double alpha, const AbstractFunctionSpaceElement &referenceState,
                                                         const FunctionSpace &domain, std::unique_ptr<Interface::AbstractOperator>&& My, std::unique_ptr<Interface::AbstractOperator>&& Mu)
    : TrackingTypeCostFunctional(alpha, referenceState, domain.sharedImpl(), std::move(My), std::move(Mu))
  {}



//  void TrackingTypeCostFunctional::setArgument(const AbstractFunctionSpaceElement &x)
//  {
//    if( isProductSpaceElement(x) ) std::cout << "Primaldualelement" << std::endl;
//    if( !isProductSpaceElement(x) ) throw InvalidArgumentException("TrackingTypeCostFunctional::setArgument");
//    AbstractFunctional::setArgument(x);
////    x_ = clone( dynamic_cast<const ProductSpaceElement&>(x) );
//  }

  double TrackingTypeCostFunctional::d0(const AbstractFunctionSpaceElement& x) const
  {
    if( !is<ProductSpaceElement>(x) ) throw InvalidArgumentException("TrackingTypeCostFunctional::operator()");

    const auto& x_ = castTo<ProductSpaceElement>(x);

    auto stateDifference = clone(*referenceState_);
    *stateDifference -= x_.variable(stateIndex());

    return 0.5 * (*(*My_)(*stateDifference)) (*stateDifference)
        + 0.5 * alpha_ * ( *(*Mu_)( x_.variable(controlIndex()) ) ) ( x_.variable(controlIndex()) );
  }

  std::unique_ptr<AbstractFunctionSpaceElement> TrackingTypeCostFunctional::d1(const AbstractFunctionSpaceElement& x) const
  {
    auto tmp = domain().element();

    const auto& x_ =  castTo<ProductSpaceElement>(x);
    auto& tmp_ = castTo<ProductSpaceElement>(*tmp);

    x_.variable(stateIndex()).copyTo( tmp_.variable(stateIndex()) );
    tmp_.variable(stateIndex()) -= *referenceState_;
    tmp_.variable(controlIndex()) = x_.variable(controlIndex());
    tmp_.variable(controlIndex()) *= alpha_;

//    return (*y)(dx);
    auto y = (*My_)(tmp_.variable(stateIndex()));
    *y   += *(*Mu_)(tmp_.variable(controlIndex()));
    return std::move(y);
//    return (*(*My_)(y_.variable(stateId_))) ( dx_.variable(stateId_) ) + (*(*Mu_)(y_.variable(controlId_))) ( dx_.variable(controlId_));
  }

  std::unique_ptr<AbstractFunctionSpaceElement> TrackingTypeCostFunctional::d2(const AbstractFunctionSpaceElement&, const AbstractFunctionSpaceElement& dx) const
  {
    if( !is<ProductSpaceElement>(dx) ) throw InvalidArgumentException("TrackingTypeCostFunctional::d2");

    auto y = clone(dx);
    auto& y_ = castTo<ProductSpaceElement>(*y);
    y_.variable(controlIndex()) *= alpha_;
    return y;
  }


//  double TrackingTypeCostFunctional::d2(const AbstractFunctionSpaceElement& dx, const AbstractFunctionSpaceElement& dy) const
//  {
//    if( !isProductSpaceElement(dx) || !isProductSpaceElement(dy) ) throw InvalidArgumentException("TrackingTypeCostFunctional::d2");

//    const auto& dx_ = dynamic_cast<const ProductSpaceElement&>(dx);
//    const auto& dy_ = dynamic_cast<const ProductSpaceElement&>(dy);

//    return dx_.variable(stateId_) * dy_.variable(stateId_)
//        + alpha_ * ( dx_.variable(controlId_) * dy_.variable(controlId_) );
//  }

  TrackingTypeCostFunctional* TrackingTypeCostFunctional::cloneImpl() const
  {
    return new TrackingTypeCostFunctional(alpha_,*referenceState_,sharedDomain(),clone(My_),clone(Mu_));
  }

}

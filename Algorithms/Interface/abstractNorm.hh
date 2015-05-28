#ifndef ALGORITHM_INTERFACE_ABSTRACT_NORM_HH
#define ALGORITHM_INTERFACE_ABSTRACT_NORM_HH

#include "../Util/callofundefinedfunctionexception.hh"

namespace Algorithm
{
  class AbstractFunctionSpaceElement;
  template <class> class Restriction;

  class AbstractNorm
  {
  public:
    virtual ~AbstractNorm(){}

    virtual double operator()(const AbstractFunctionSpaceElement&) const = 0;

    virtual double operator()(const Restriction<AbstractFunctionSpaceElement>&)
    {
      throw CallOfUndefinedFunctionException("AbstractNorm::operator()(const Restriction&)");
    }

    virtual double squared(const AbstractFunctionSpaceElement&) const = 0;
  };
}

#endif // ALGORITHM_INTERFACE_ABSTRACT_NORM_HH

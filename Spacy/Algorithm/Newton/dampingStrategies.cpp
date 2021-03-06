#include "dampingStrategies.hh"

#include <Spacy/vectorSpace.hh>
#include <Spacy/zeroVectorCreator.hh>
#include <Spacy/c1Operator.hh>
#include <Spacy/Util/Exceptions/regularityTestFailedException.hh>

#include <cmath>

namespace Spacy
{
  namespace Newton
  {
    namespace Damping
    {
      AffineCovariant::AffineCovariant(const C1Operator& F)
        : F_(F), oldDs(zero(F.domain()))
      {}

      DampingFactor AffineCovariant::operator()(const std::function<Vector(const Vector&)>& DFInv_, const Vector& x, const Vector& dx) const
      {
        auto nu = DampingFactor{1};
        auto mu = 1.;
        auto normDx =  norm(dx);

        if( normDx < sqrt_eps() * norm(x) )
            return nu;

        if( oldNu > 0 )
        {
          mu = get(normOldDx * normOldDs * oldNu / ( normDx * norm(oldDs - dx) ));
          nu = std::min(1.,mu);
        }

        while(true)
        {
          if( regularityTestFailed(nu) )
              throw RegularityTestFailedException("Newton::DampingStrategy::AffineCovariant",get(get(nu)));

          auto trial = x + nu*dx;
          auto ds = DFInv_(-F_(trial)) - (1-nu)*dx;
          auto normDs = norm(ds);

          auto muPrime = DampingFactor{0.5*nu*nu/normDs};

          if( normDs/normDx >= 1)
          {
            nu = min(0.5*nu,muPrime);
            continue;
          }

          auto nuPrime = min(1.,muPrime);

          if( nu == 1 && nuPrime == 1 && normDs < eps() )
              break;

          if( nuPrime >= 4*nu)
          {
              nu = 4*nu;
            // The following line is the damping strategy according Deuflhard.
            // To avoid cycling the above heuristic is employed.
//            nu = nuPrime;
            continue;
          }

          break;
        }

        return nu;
      }


      AffineContravariant::AffineContravariant(const C1Operator& F)
        : F_(F)
      {}

      DampingFactor AffineContravariant::operator()(const std::function<Vector(const Vector&)>&, const Vector& x, const Vector& dx) const
      {
        auto nu = DampingFactor(1);
        auto norm_F_x = norm(F_(x));
        if( norm_F_x < sqrt_eps() )
            return nu;

        if( muPrime > 0 )
          nu = min( 1. , muPrime*norm_F_x_old/norm_F_x );

        while( true )
        {
          if( !regularityTestPassed(nu))
              throw RegularityTestFailedException("Newton::DampingStrategy::AffineContravariant",get(get(nu)));

          auto trial = x + nu*dx;

          auto norm_F_trial = norm(F_(trial));

          auto theta = norm_F_trial/norm_F_x;
          muPrime = 0.5*norm_F_x*nu*nu / norm( F_(trial) - (1-nu)*F_(x) );

          if( theta >= 1 )
          {
            nu = min( muPrime , get(0.5*nu) );
            continue;
          }

          auto oldNu = nu;
          nu = min( 1. , muPrime );
          if( nu >= 4*oldNu )
          {
            nu = 4*oldNu;
            continue;
          }

          norm_F_x_old = norm_F_x;
          break;
        }

        return nu;
      }


      None::None(const C1Operator&)
      {}

      DampingFactor None::operator()(const std::function<Vector(const Vector&)>&, const Vector&, const Vector&) const
      {
        return DampingFactor(1);
      }
    }
  }
}

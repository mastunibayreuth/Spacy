#pragma once 

#include <dune/common/fvector.hh>

namespace Kaskade
{
    
    
    
  template <class ReferenceSolution, class Varset, int stateId, int controlId>
  class TrackingFunctional
  {
  public:
      // Reference Solution and Varset different
    typedef typename Varset::Descriptions AnsatzVars;
    typedef typename Varset::Descriptions::Scalar Scalar;
    static constexpr int dim = AnsatzVars::Grid::dimension;

    static int const stateSpaceId = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables,stateId>::type::spaceIndex;

    template <class Evaluators>
    void evaluateAt(Dune::FieldVector<Scalar, AnsatzVars::template Components<stateId>::m> const& y_,
                    Dune::FieldVector<Scalar, AnsatzVars::template Components<controlId>::m> const& u_,
                    Evaluators const& evaluators)
    {
      y = y_;
      u = u_;
      y_ref = boost::fusion::at_c<stateId>(y_ref_fse.data).value(boost::fusion::at_c<stateSpaceId>(evaluators));
    }

    Scalar d0(int id = -1) const
    {
      auto y_z = y - y_ref;
      if(id == stateId)
      {
          return 0.5 * beta * y_z*y_z;
      }
      
      if(id == controlId) 
          {   
              return 0.5 * alpha * u * u;
          }
          
      return 0.5*beta*y_z * y_z + 0.5 * alpha * u * u;
    }

    template<int row>
    Scalar d1 (VariationalArg<Scalar,dim,AnsatzVars::template Components<row>::m> const& arg) const
    {
      if(row==stateId) return beta*(y - y_ref) * arg.value;
      if(row==controlId) return alpha*(u*arg.value);
      return 0;
    }

    template<int row, int col>
    Scalar d2 (VariationalArg<Scalar,dim,AnsatzVars::template Components<row>::m> const& arg1,
               VariationalArg<Scalar,dim,AnsatzVars::template Components<col>::m> const& arg2) const
    {
      if(row==stateId && col==stateId) return beta*arg1.value*arg2.value;
      if(row==controlId && col==controlId) return alpha*arg1.value*arg2.value;
      return 0;
    }

    TrackingFunctional(Scalar alpha_, ReferenceSolution const& fse, Scalar beta_=1.0) : alpha(alpha_), beta(beta_) , y_ref_fse(fse), y_ref(0), y(0), u(0)
    {
//       std::cout <<  "TrackingFunctional" <<  std::endl;
      assert(alpha > 0);
    }

    TrackingFunctional(TrackingFunctional const&) = default;
    TrackingFunctional& operator=(TrackingFunctional const&) = default;

    void setReferenceSolution(ReferenceSolution const& fse)
    {
      y_ref_fse = fse;
    }

  private:
    Scalar alpha, beta;
    ReferenceSolution y_ref_fse;
    Dune::FieldVector<Scalar,AnsatzVars::template Components<stateId>::m> y_ref, y;
    Dune::FieldVector<Scalar,AnsatzVars::template Components<controlId>::m> u;
  };

}

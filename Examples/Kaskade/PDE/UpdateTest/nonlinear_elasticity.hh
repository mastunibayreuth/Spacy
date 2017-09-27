#pragma once
#include <iostream>
#include <fstream>
#include "../../traits.hh"
#include "../../boundary_caches.hh"
#include "domain_caches.hh"

// #include <fem/functional_aux.hh>

#include <algorithm>

namespace Kaskade
{
template <class Integrand, class VarSet, int state_index = 0>
class NonlinearElasticityFunctional: public FunctionalBase<VariationalFunctional>
{
public:
    using Scalar = double;
    using OriginVars = VarSet;
    using AnsatzVars = VarSet;
    using TestVars = VarSet;
    static const int dim = AnsatzVars::Grid::dimension;
    using Vector = Dune::FieldVector<Scalar, dim>;
    const Vector surface_force_density_;
    static int constexpr u_Idx = 0;
    static int constexpr u_Space_Idx = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables, u_Idx>::type::spaceIndex;

    template <int row>
    using D1 = NonConstD1<row>;

    template <int row, int col>
    using D2 = D2PresentAndNonSymmetric<row,col>;

    using DomainCache = FunctionalWithConstantSource<NonlinearElasticityFunctional, Integrand, state_index>;

    class BoundaryCache : public CacheBase<NonlinearElasticityFunctional,BoundaryCache>
    {
    public:
        using FaceIterator = typename AnsatzVars::Grid::LeafIntersectionIterator;

        BoundaryCache(NonlinearElasticityFunctional const& f_, typename AnsatzVars::VariableSet const& vars_, int flags=7)
            : vars(vars_), force_density_(f_.surface_force_density_), beta(0.), up {0, 0, 1},  side {0,  1, 0}
        {}


        void moveTo(FaceIterator const& face)
        {
            e = &face;

            const auto scalarProdUp = face->centerUnitOuterNormal() *up;
            const auto scalarSide = face->centerUnitOuterNormal() *side;

            if (scalarProdUp > 0.5)
            {
            	alpha = 0.0;
                 beta = 0.0;
            }

            else if (scalarProdUp < -0.5)
            {
                alpha = 0.0;
                beta  = force_density_;
//                std::cout << "ForceDensity: " << force_density_ << std::endl;
            }

//            else if(scalarSide < -0.5)
//            {
//                alpha = dirichlet_penalty;
//                beta  = 0.0;
//            }
            
            else
            {
            	alpha =dirichlet_penalty;
            	                 beta = 0.0;
            }


        }
        
        template <class Evaluators>
        void evaluateAt(Dune::FieldVector<typename AnsatzVars::Grid::ctype,dim-1> const& x, Evaluators const& evaluators) 
        {
           using namespace boost::fusion;
           glob_ = (*e)->geometry().global(x);
           u = at_c<u_Idx>(vars.data).value(at_c<u_Space_Idx>(evaluators));
        }
         
//         template <class Evaluators>
//         void evaluateAt(Dune::FieldVector<typename AnsatzVars::Grid::ctype,dim-1> const& x, Evaluators const& evaluators )
//         {
//             using namespace boost::fusion;
//             u = at_c<u_Idx>(vars.data).value(at_c<u_Space_Idx>(evaluators));
//         }

        Scalar
        d0() const
        {
            return 0.5*alpha*((u)*(u)) - beta*(u+glob_);
        }

        template<int row>
        Scalar d1_impl (VariationalArg<Scalar,dim,dim> const& arg) const
        {
            return alpha*(u*arg.value) - beta*(arg.value/*+loc_*/);
        }

        template<int row, int col>
        Scalar d2_impl (VariationalArg<Scalar,dim,dim> const &arg1, VariationalArg<Scalar,dim,dim> const &arg2) const
        {
            return alpha*((arg1.value) *(arg2.value));
        }

        static constexpr Scalar dirichlet_penalty = 1e11;
    private:
        typename AnsatzVars::VariableSet const& vars;
        const Vector & force_density_;
        Vector u, beta;
        const Vector up, side;
        Scalar alpha = 0.0;
        FaceIterator const* e = nullptr;
        Dune::FieldVector<typename AnsatzVars::Grid::ctype,dim> glob_ = {0.0};


    };

    Integrand f_;

    explicit NonlinearElasticityFunctional(Integrand f, const Vector & surface_force_density) : f_(std::move(f)), surface_force_density_(surface_force_density)
    {}

    template <class Cell>
    int integrationOrder(Cell const& /*cell*/, int shapeFunctionOrder, bool boundary) const
    {
        if (boundary)
            return 2*shapeFunctionOrder;
        else
        {
            int stiffnessMatrixIntegrationOrder = 2*(shapeFunctionOrder-1);
            int sourceTermIntegrationOrder = shapeFunctionOrder;
            return std::max(stiffnessMatrixIntegrationOrder,sourceTermIntegrationOrder);
        }
    }
};
    template<typename Matrix>
    void writeMatlab(const Matrix & matrix)
    {
      std::ofstream myfile;
      myfile.open ("matrix.dat");
      for (int i = 0; i < matrix.ridx.size(); i++)
      {    
        myfile << matrix.ridx[i]+1;
        myfile << " ";
        myfile << matrix.cidx[i]+1;
        myfile << " ";
        myfile << matrix.data[i];
        myfile << std::endl;
      }
      myfile.close();
    }

}

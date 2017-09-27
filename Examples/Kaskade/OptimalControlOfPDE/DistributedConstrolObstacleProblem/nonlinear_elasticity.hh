/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*  This file is part of the library KASKADE 7                               */
/*    see http://www.zib.de/Numerik/numsoft/kaskade7/                        */
/*                                                                           */
/*  Copyright (C) 2002-2009 Zuse Institute Berlin                            */
/*                                                                           */
/*  KASKADE 7 is distributed under the terms of the ZIB Academic License.    */
/*    see $KASKADE/academic.txt                                              */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#pragma once

#include <memory>
#include <type_traits>

#include "fem/variables.hh"
#include "utilities/linalg/scalarproducts.hh"
#include "trackingFunctional.hh"
#include "functional.hh"
// #include "fem/diffops/trackingTypeCostFunctional.hh"
// #include "fem/diffops/antonsNonlinearTestProblems.hh"


/// \cond
using namespace Kaskade;


/****************************************************************************************/
/* Boundary */
/****************************************************************************************/
enum class RoleOfFunctional
{
    NORMAL, TANGENTIAL, PRECONDITIONING
};

// Reference Deformation has to be template ??
template <class Integrand, class Reference, int stateId, int controlId, int adjointId, class RType, class AnsatzVars_, class TestVars_=AnsatzVars_, class OriginVars_=AnsatzVars_, RoleOfFunctional role = RoleOfFunctional::NORMAL, bool lump=false >
class StepFunctional
{
public:
    typedef RType  Scalar;
    typedef AnsatzVars_ AnsatzVars;
    typedef TestVars_ TestVars;
    typedef OriginVars_ OriginVars;
    static constexpr int dim = AnsatzVars::Grid::dimension;
    static ProblemType const type = std::is_same<AnsatzVars,TestVars>::value ? VariationalFunctional : WeakFormulation;

    typedef typename AnsatzVars::Grid Grid;
    typedef typename Grid::template Codim<0>::Entity Cell;

    static int const yIdx = stateId;
    static int const uIdx = controlId;
    static int const pIdx = adjointId;

    static int const ySIdx = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables,yIdx>::type::spaceIndex;
    static int const uSIdx = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables,uIdx>::type::spaceIndex;
    static int const pSIdx = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables,pIdx>::type::spaceIndex;

    class DomainCache : public CacheBase<StepFunctional,DomainCache>
    {
    public:
        DomainCache(StepFunctional const& f_,
                    typename AnsatzVars::VariableSet const& vars_,int flags=7):
            f(f_), vars(vars_), c(f.integrand_), J(f.J)
        {}

        template <class Position, class Evaluators>
        void evaluateAt(Position const& x, Evaluators const& evaluators)
        {

        	// check what is necessary Do no evaluate twice!!
            using namespace boost::fusion;
            y = at_c<yIdx>(vars.data).value(at_c<ySIdx>(evaluators));
//             if ( y*y > 10 ) std::cout << "Y: " << y*y << std::endl; 
            u = at_c<uIdx>(vars.data).value(at_c<uSIdx>(evaluators));
            // check what is really necessary
            p.value = at_c<pIdx>(vars.data).value(at_c<pSIdx>(evaluators));

            p.derivative = at_c<pIdx>(vars.data).derivative(at_c<pSIdx>(evaluators));

            c.evaluateAt(vars, evaluators);

           /* const auto value = c.d0()*/;
//            std::cout << "valuetest: " << value << '\n';

            J.evaluateAt(y,u,evaluators);
        }

        // Why J.0() ??
        Scalar d0() const
        {
            return J.d0(stateId);
        }

        template<int row>
        Scalar d1_impl (VariationalArg<Scalar,dim, TestVars::template Components<row>::m> const& arg) const
        {
            // merge control into c   //Change postion of p and arg
            if(row==yIdx) return J.template d1<yIdx>(arg) + c.template d2<yIdx,yIdx>(p,arg);

            if(row==pIdx) return c.template d1<yIdx>(arg) /* - u*if_(arg.value,p.value)*/;

            return 0;
        }

        template<int row, int col>
        Scalar d2_impl (VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const &arg1, VariationalArg<Scalar,dim,TestVars::template Components<col>::m> const &arg2) const
        {
            if(row == yIdx && col == yIdx)
            {
                if(role == RoleOfFunctional::TANGENTIAL )
                    return J.template d2<yIdx,yIdx>(arg1,arg2) + c.template d3<yIdx,yIdx,yIdx>(p,arg1,arg2);
                else
                {
                    if(role == RoleOfFunctional::NORMAL )
                    {
                        return J.template d2<yIdx,yIdx>(arg1,arg2);
                    }
                }
            }

            // check
            if(row == yIdx && col == pIdx) return c.template d2<yIdx,yIdx>(arg2,arg1);
            if(row == pIdx && col == yIdx) return c.template d2<yIdx,yIdx>(arg1,arg2);
            return 0;
        }

    private:
        StepFunctional const& f;
        typename AnsatzVars::VariableSet const& vars;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<yIdx>::m> y, y_z;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<uIdx>::m> u;
        NonlinearElasticityFunctionalControl<Integrand,  AnsatzVars,yIdx> c;
        TrackingFunctional<Reference,  typename AnsatzVars::VariableSet,yIdx,uIdx> J;
        // Why not a FieldVector ??
        VariationalArg<Scalar,dim,AnsatzVars::template Components<pIdx>::m> p;

    };

    class BoundaryCache : public CacheBase<StepFunctional,BoundaryCache>
    {
    public:
        using FaceIterator = typename AnsatzVars::Grid::LeafIntersectionIterator;
        using Vector = Dune::FieldVector<Scalar, dim>;
        BoundaryCache(StepFunctional const& f,
                      typename AnsatzVars::VariableSet const& vars_, int flags=7):
            vars(vars_), dirichlet_penalty(f.gamma),  J(f.J),   up {0, 0, 1},  side {0,  1, 0}, localPenalty_(0.0), normalComplianceParameter_(f.normalComplianceParameter_), obstacle_(f.obstacle_)
        {
            std::cout << "boundaryCacheNormalComplianceParameter: " << normalComplianceParameter_ << std::endl << std::endl;
        }

            void moveTo(FaceIterator const& face)
                    {

            	const auto scalarProdUp = face->centerUnitOuterNormal() *up;
               // const auto scalarProdSide = face->centerUnitOuterNormal() *side;
                // change order for more efficiency
                //
                        if (scalarProdUp > 0.5)
                        {
                            alpha = 0.0;
                            localPenalty_ = 0.0;
                             beta  = 1.0;
                        }
                        else if (scalarProdUp < -0.5)
                        {
                            alpha = 0.0;
                            beta  = 0.0;
                            localPenalty_ = normalComplianceParameter_;
                        }

//                        else if (scalarProdSide > 0.5 ||  scalarProdSide < -0.5)
//                        {
//                            alpha = dirichlet_penalty;
//                            beta  = 0.0;
//                            localPenalty_ = 0.0;
//                        }

                        else
                        {
//                            alpha = 0.0;
                        	 alpha = dirichlet_penalty;
                            beta = 0.0;
                            localPenalty_ = 0.0;
                        }
                    }


        template <class Evaluators>
        void evaluateAt(Dune::FieldVector<typename AnsatzVars::Grid::ctype,AnsatzVars::Grid::dimension-1> const& x, Evaluators const& evaluators)
        {
            using namespace boost::fusion;
            y = at_c<yIdx>(vars.data).value(at_c<ySIdx>(evaluators));
            u = at_c<uIdx>(vars.data).value(at_c<uSIdx>(evaluators));
            p = at_c<pIdx>(vars.data).value(at_c<pSIdx>(evaluators));


            J.evaluateAt(y,u,evaluators);

            if (localPenalty_ > 0. && (obstacle_ - y[2])  < 0.)                             //works only when coord are 0 !!
                       {

                       	localPenalty_ = 0.0;
                       }
        }

        Scalar d0() const
        { 
            // maybe do not need beta 
           // use data from J and c
            // more efficient using if conditions and beta is bool
            return beta*J.d0(controlId);
        }

        template<int row>
        Scalar d1_impl (VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const& arg) const
        {
        	// c' = I''
            if(row == yIdx) return (alpha*arg.value) * p + localPenalty_* 2.0 * (obstacle_ - y[2]) * p[2] * arg.value[2];
            if(row == uIdx) return  beta *( J.template d1<uIdx>(arg) -p*arg.value);
            if(row == pIdx) return alpha*y*arg.value - (beta* u*arg.value) - (localPenalty_ * (obstacle_ - y[2]) * (obstacle_ - y[2]) * arg.value[2]);

            return 0;
        }

        template<int row, int col>
        Scalar d2_impl (VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const &arg1, VariationalArg<Scalar,dim,AnsatzVars::template Components<col>::m> const &arg2) const
        {

        	if(row == yIdx && col == yIdx)
        	            {
        		// check again  dirichlet term missing // is correct
        	                if(role == RoleOfFunctional::TANGENTIAL )
        	                	return - 2.0 * (arg2.value[2])* localPenalty_ * p[2] * arg1.value[2];
        	                else
        	                {
        	                    if(role == RoleOfFunctional::NORMAL )
        	                    {
        	                        return 0.0;
        	                    }
        	                }
        	            }


            if (row == pIdx && col == pIdx)  return 0;
            if (row==yIdx && col==pIdx) return alpha*(arg1.value * arg2.value) + (localPenalty_ * 2.0 * (obstacle_ - y[2]) * arg1.value[2] *arg2.value[2]);

            if(row==pIdx && col==yIdx)  return alpha*(arg1.value * arg2.value) + (localPenalty_ * 2.0 * (obstacle_ - y[2]) * arg1.value[2] *arg2.value[2]);
            if(row == uIdx && col == uIdx) return beta* J.template d2<uIdx,uIdx>(arg1,arg2);

            if(row==pIdx && col==uIdx) return - beta * arg1.value * arg2.value;
            if(row==uIdx && col==pIdx) return - beta * arg2.value * arg1.value;

            return 0;
        }

    private:
        typename AnsatzVars::VariableSet const& vars;
//         Dune::FieldVector<Scalar,AnsatzVars::template Components<yIdx>::m> u;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<uIdx>::m> u;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<yIdx>::m> y;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<pIdx>::m> p;

        // use Reference
        TrackingFunctional<Reference,  typename AnsatzVars::VariableSet,  yIdx,uIdx> J;

        const Vector up, side;

        FaceIterator const* e;
        double alpha, beta;
        const double dirichlet_penalty;
        double normalComplianceParameter_;
        double localPenalty_;
        const double obstacle_;
    };

    // use std::move by copy übergabe !!!!!  // alpha J and beta ??
    explicit StepFunctional(Scalar regularization, Reference const& ref,  Integrand integrand, double normalComplianceParameter, double obstacle) :
        gamma(1e9), alpha(0.), J(regularization,ref), integrand_(integrand), normalComplianceParameter_(normalComplianceParameter),obstacle_(obstacle)
    {

    }

    template <class T>
    bool inDomain(T const&) const
    {
        return true;
    }

    template <int row>
    struct D1 : public FunctionalBase<WeakFormulation>::D1<row>
    {
        static bool const present = true;
        static bool const constant = false;
    };

    template <int row, int col>
    struct D2 : public FunctionalBase<WeakFormulation>::D2<row,col>
    {
        static bool const present = !( (row==yIdx && col==yIdx && role==RoleOfFunctional::PRECONDITIONING ) ||
                                       ( row == pIdx && col == pIdx ) ||
                                       ( row == yIdx && col == uIdx ) ||
                                       ( row == uIdx && col == yIdx ) );
        static bool const symmetric = row==col;
        static bool const lumped =  (row==uIdx && col==uIdx && role==RoleOfFunctional::PRECONDITIONING );
    };

    template <class Cell>
    int integrationOrder(Cell const& /* cell */, int shapeFunctionOrder, bool  boundary ) const
    {
        if( boundary ) return 2*shapeFunctionOrder;
        return 4*shapeFunctionOrder - 2;
    }

    void paramUpdate(double update)
    {
    	// std::cout << "UpdateComplianceParameter: " <<  "BeforeUpdate: " << normalComplianceParameter_ << "AfterUpdate: " <<  update << '\n';
    	normalComplianceParameter_ = update;
    }

    Scalar gamma, alpha;
    TrackingFunctional<Reference, typename AnsatzVars::VariableSet, yIdx,uIdx> J;
    // Functional C constraints
    Integrand integrand_;
    double normalComplianceParameter_;
    const double obstacle_;
};


template <class Integrand, class Reference, int stateId, int controlId, int adjointId, class RType, class AnsatzVars, class TestVars=AnsatzVars, class OriginVars=AnsatzVars>
using NormalStepFunctional = StepFunctional<Integrand, Reference, stateId, controlId,adjointId,RType,AnsatzVars,TestVars,OriginVars,RoleOfFunctional::NORMAL>;

template <class Integrand, class Reference,  int stateId, int controlId, int adjointId, class RType, class AnsatzVars, class TestVars=AnsatzVars, class OriginVars=AnsatzVars>
using TangentialStepFunctional = StepFunctional<Integrand, Reference, stateId,controlId,adjointId,RType,AnsatzVars,TestVars,OriginVars,RoleOfFunctional::TANGENTIAL>;

template <class Integrand, class Reference, int stateId, int controlId, int adjointId, class RType, class AnsatzVars_, class TestVars_=AnsatzVars_, class OriginVars_=AnsatzVars_, RoleOfFunctional role = RoleOfFunctional::NORMAL, bool lump=false >
class TestFunctional
{
public:
    typedef RType  Scalar;
    typedef AnsatzVars_ AnsatzVars;
    typedef TestVars_ TestVars;
    typedef OriginVars_ OriginVars;
    static constexpr int dim = AnsatzVars::Grid::dimension;
    static ProblemType const type = std::is_same<AnsatzVars,TestVars>::value ? VariationalFunctional : WeakFormulation;

    typedef typename AnsatzVars::Grid Grid;
    typedef typename Grid::template Codim<0>::Entity Cell;

    static int const yIdx = stateId;
    static int const uIdx = controlId;
    static int const pIdx = adjointId;

    static int const ySIdx = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables,yIdx>::type::spaceIndex;
    static int const uSIdx = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables,uIdx>::type::spaceIndex;
    static int const pSIdx = boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables,pIdx>::type::spaceIndex;

    class DomainCache : public CacheBase<TestFunctional,DomainCache>
    {
    public:
        DomainCache(TestFunctional const& f_,
                    typename AnsatzVars::VariableSet const& vars_,int flags=7):
                f(f_), vars(vars_), c(f.integrand_), J(f.J)
        {}

        template <class Position, class Evaluators>
        void evaluateAt(Position const& x, Evaluators const& evaluators)
        {

            // check what is necessary Do no evaluate twice!!
            using namespace boost::fusion;
            y = at_c<yIdx>(vars.data).value(at_c<ySIdx>(evaluators));
//             if ( y*y > 10 ) std::cout << "Y: " << y*y << std::endl;
            u = at_c<uIdx>(vars.data).value(at_c<uSIdx>(evaluators));
            // check what is really necessary
            p.value = at_c<pIdx>(vars.data).value(at_c<pSIdx>(evaluators));

            p.derivative = at_c<pIdx>(vars.data).derivative(at_c<pSIdx>(evaluators));

            c.evaluateAt(vars, evaluators);

            /* const auto value = c.d0()*/;
//            std::cout << "valuetest: " << value << '\n';

            J.evaluateAt(y,u,evaluators);
        }

        // Why J.0() ??
        Scalar d0() const
        {
            return J.d0(stateId);
        }

        template<int row>
        Scalar d1_impl (VariationalArg<Scalar,dim, TestVars::template Components<row>::m> const& arg) const
        {
            // merge control into c   //Change postion of p and arg


            return 0;
        }

        template<int row, int col>
        Scalar d2_impl (VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const &arg1, VariationalArg<Scalar,dim,TestVars::template Components<col>::m> const &arg2) const
        {

            return 0;
        }

    private:
        TestFunctional const& f;
        typename AnsatzVars::VariableSet const& vars;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<yIdx>::m> y, y_z;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<uIdx>::m> u;
        NonlinearElasticityFunctionalControl<Integrand,  AnsatzVars,yIdx> c;
        TrackingFunctional<Reference,  typename AnsatzVars::VariableSet,yIdx,uIdx> J;
        // Why not a FieldVector ??
        VariationalArg<Scalar,dim,AnsatzVars::template Components<pIdx>::m> p;

    };

    class BoundaryCache : public CacheBase<TestFunctional,BoundaryCache>
    {
    public:
        using FaceIterator = typename AnsatzVars::Grid::LeafIntersectionIterator;
        using Vector = Dune::FieldVector<Scalar, dim>;
        BoundaryCache(TestFunctional const& f,
                      typename AnsatzVars::VariableSet const& vars_, int flags=7):
                vars(vars_), dirichlet_penalty(f.gamma),  J(f.J),   up {0, 0, 1},  side {0,  1, 0}, localPenalty_(0.0), normalComplianceParameter_(f.normalComplianceParameter_), obstacle_(f.obstacle_)
        {}

        void moveTo(FaceIterator const& face)
        {

            const auto scalarProdUp = face->centerUnitOuterNormal() *up;
            // const auto scalarProdSide = face->centerUnitOuterNormal() *side;
            // change order for more efficiency
            //
            if (scalarProdUp > 0.5)
            {
                alpha = 0.0;
                localPenalty_ = 0.0;
                beta  = 1.0;
            }
            else if (scalarProdUp < -0.5)
            {
                alpha = 0.0;
                beta  = 0.0;
                localPenalty_ = normalComplianceParameter_;
            }

//                        else if (scalarProdSide > 0.5 ||  scalarProdSide < -0.5)
//                        {
//                            alpha = dirichlet_penalty;
//                            beta  = 0.0;
//                            localPenalty_ = 0.0;
//                        }

            else
            {
//                            alpha = 0.0;
                alpha = dirichlet_penalty;
                beta = 0.0;
                localPenalty_ = 0.0;
            }
        }


        template <class Evaluators>
        void evaluateAt(Dune::FieldVector<typename AnsatzVars::Grid::ctype,AnsatzVars::Grid::dimension-1> const& x, Evaluators const& evaluators)
        {
            using namespace boost::fusion;
            y = at_c<yIdx>(vars.data).value(at_c<ySIdx>(evaluators));
            u = at_c<uIdx>(vars.data).value(at_c<uSIdx>(evaluators));
            p = at_c<pIdx>(vars.data).value(at_c<pSIdx>(evaluators));


            J.evaluateAt(y,u,evaluators);

            if (localPenalty_ > 0. && (obstacle_ - y[2])  < 0.)                             //works only when coord are 0 !!
            {

                localPenalty_ = 0.0;
            }
        }

        Scalar d0() const
        {
            // maybe do not need beta
            // use data from J and c
            // more efficient using if conditions and beta is bool
            return beta*J.d0(controlId);
        }

        template<int row>
        Scalar d1_impl (VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const& arg) const
        {
            // c' = I''
            if(row == yIdx) return localPenalty_* 2.0 * (obstacle_ - y[2]) * p[2] * arg.value[2];
            if(row == uIdx) return 0.0;
            if(row == pIdx) return  -(localPenalty_ * (obstacle_ - y[2]) * (obstacle_ - y[2]) * arg.value[2]);

            return 0;
        }

        template<int row, int col>
        Scalar d2_impl (VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const &arg1, VariationalArg<Scalar,dim,AnsatzVars::template Components<col>::m> const &arg2) const
        {

            if(row == yIdx && col == yIdx)
            {
                // check again  dirichlet term missing // is correct
                if(role == RoleOfFunctional::TANGENTIAL )
                    return - 2.0 * (arg2.value[2])* localPenalty_ * p[2] * arg1.value[2];
                else
                {
                    if(role == RoleOfFunctional::NORMAL )
                    {
                        return 0.0;
                    }
                }
            }


            if (row == pIdx && col == pIdx)  return 0;
            if (row==yIdx && col==pIdx) return alpha*(arg1.value * arg2.value) + (localPenalty_ * 2.0 * (obstacle_ - y[2]) * arg1.value[2] *arg2.value[2]);

            if(row==pIdx && col==yIdx)  return alpha*(arg1.value * arg2.value) + (localPenalty_ * 2.0 * (obstacle_ - y[2]) * arg1.value[2] *arg2.value[2]);
            if(row == uIdx && col == uIdx) return beta* J.template d2<uIdx,uIdx>(arg1,arg2);

            if(row==pIdx && col==uIdx) return - beta * arg1.value * arg2.value;
            if(row==uIdx && col==pIdx) return - beta * arg2.value * arg1.value;

            return 0;
        }

    private:
        typename AnsatzVars::VariableSet const& vars;
//         Dune::FieldVector<Scalar,AnsatzVars::template Components<yIdx>::m> u;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<uIdx>::m> u;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<yIdx>::m> y;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<pIdx>::m> p;

        // use Reference
        TrackingFunctional<Reference,  typename AnsatzVars::VariableSet,  yIdx,uIdx> J;

        const Vector up, side;

        FaceIterator const* e;
        double alpha, beta;
        const double dirichlet_penalty;
        double normalComplianceParameter_;
        double localPenalty_;
        const double obstacle_;
    };

    // use std::move by copy übergabe !!!!!  // alpha J and beta ??
    explicit TestFunctional (Scalar regularization, Reference const& ref,  Integrand integrand, double normalComplianceParameter, double obstacle) :
            gamma(1e9), alpha(0.), J(regularization,ref), integrand_(integrand), normalComplianceParameter_(normalComplianceParameter),obstacle_(obstacle)
    {

    }

    template <class T>
    bool inDomain(T const&) const
    {
        return true;
    }

    template <int row>
    struct D1 : public FunctionalBase<WeakFormulation>::D1<row>
    {
        static bool const present = true;
        static bool const constant = false;
    };

    template <int row, int col>
    struct D2 : public FunctionalBase<WeakFormulation>::D2<row,col>
    {
        static bool const present = !( (row==yIdx && col==yIdx && role==RoleOfFunctional::PRECONDITIONING ) ||
                                       ( row == pIdx && col == pIdx ) ||
                                       ( row == yIdx && col == uIdx ) ||
                                       ( row == uIdx && col == yIdx ) );
        static bool const symmetric = row==col;
        static bool const lumped =  (row==uIdx && col==uIdx && role==RoleOfFunctional::PRECONDITIONING );
    };

    template <class Cell>
    int integrationOrder(Cell const& /* cell */, int shapeFunctionOrder, bool  boundary ) const
    {
        if( boundary ) return 2*shapeFunctionOrder;
        return 4*shapeFunctionOrder - 2;
    }

    void paramUpdate(double update)
    {
        std::cout << "UpdateComplianceParameter: " <<  "BeforeUpdate: " << normalComplianceParameter_ << "     AfterUpdate: " <<  update << '\n';
        normalComplianceParameter_ = update;
    }

    Scalar gamma, alpha;
    TrackingFunctional<Reference, typename AnsatzVars::VariableSet, yIdx,uIdx> J;
    // Functional C constraints
    Integrand integrand_;
    double normalComplianceParameter_;
    const double obstacle_;
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


/// \endcond


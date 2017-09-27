#pragma once

#include <dune/common/fvector.hh>
#include <fem/functional_aux.hh>
#include <utilities/linalg/scalarproducts.hh>
#include <utilities/linalg/determinant.hh>

#include <tuple>

namespace Kaskade
{
    template <class Operator, class Integrand, int state_index>
    class OperatorWithConstantSource
            : public CacheBase< Operator, OperatorWithConstantSource<Operator, Integrand, state_index> >
    {
        using Scalar = typename Operator::Scalar;
        using AnsatzVars = typename Operator::AnsatzVars;
        using TestVars = typename Operator::TestVars;

        static constexpr int dim = AnsatzVars::Grid::dimension;

        static constexpr int state_space_index =
                boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables, state_index>::type::spaceIndex;

        template <int entry>
        using AnsatzVarArg = VariationalArg<Scalar, dim, AnsatzVars::template Components<entry>::m>;

        template <int entry>
        using TestVarArg = VariationalArg<Scalar, dim, TestVars::template Components<entry>::m>;


    public:
        OperatorWithConstantSource(const Operator& F,
                              typename AnsatzVars::VariableSet const& x,
                              int):
            f_(F.f_), x_(x), g_(1)
        {}

        template <class Position, class Evaluators>
        void evaluateAt(const Position&, Evaluators const& evaluators)
        {
            using boost::fusion::at_c;

            f_.template update<state_index>( std::make_tuple(
                                                 at_c<state_index>(x_.data).value(at_c<state_space_index>(evaluators)),
                                                 at_c<state_index>(x_.data).gradient(at_c<state_space_index>(evaluators))
                                                 ) );
        }

        Scalar
        d0() const
        {
            return 0;
        }

        template<int row>
        Scalar d1_impl (const TestVarArg<row>& arg) const
        {
            return sp( f_(), arg.gradient ) - sp( g_, arg.value );
        }

        template<int row, int col>
        Scalar d2_impl (const TestVarArg<row>& arg1, const AnsatzVarArg<col>& arg2) const
        {
            return sp( f_.template d1<state_index>( std::make_tuple(arg2.value, arg2.gradient) ), arg1.gradient );
        }

    private:
        Integrand f_;
        const typename AnsatzVars::VariableSet& x_;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<state_index>::m> g_;
        LinAlg::EuclideanScalarProduct sp;
    };


    template <class Functional, class Integrand, int state_index>
    class FunctionalWithConstantSource
            : public CacheBase< Functional, FunctionalWithConstantSource<Functional, Integrand, state_index> >
    {
        using Scalar = typename Functional::Scalar;
        using AnsatzVars = typename Functional::AnsatzVars;
        using TestVars = typename Functional::TestVars;

        static constexpr int dim = AnsatzVars::Grid::dimension;

        static constexpr int state_space_index =
                boost::fusion::result_of::value_at_c<typename AnsatzVars::Variables, state_index>::type::spaceIndex;

        template <int entry>
        using AnsatzVarArg = VariationalArg<Scalar, dim, AnsatzVars::template Components<entry>::m>;

        template <int entry>
        using TestVarArg = VariationalArg<Scalar, dim, TestVars::template Components<entry>::m>;


    public:
        FunctionalWithConstantSource(const Functional& F,
                              typename AnsatzVars::VariableSet const& x,
                              int):
            f_(F.f_), x_(x),id_(0.0), y_(0), g_(0)
          {id_[0][0] = id_[1][1] = id_[2][2] = 1.0;}

        template<class Entity>
        void moveTo(const Entity & entity)
        {
        	e = &entity;
        }

        template <class Position, class Evaluators>
        void evaluateAt(const Position&, Evaluators const& evaluators)
        {
            using boost::fusion::at_c;

// do no use rvalues for updates!!  //Strain Tensor needs displacements !!!!!!!
           // y_ = at_c<state_index>(x_.data).value(at_c<state_space_index>(evaluators));
//            dy_ = at_c<state_index>(x_.data).derivative(at_c<state_space_index>(evaluators));
//            dy_+=id_;
//            dy_.invert();

            //dy_ += id_;
            //Determinant<3> det(dy_);
            //auto value = det();
            //if(value <= 0)
            //{
//            	std::cout << "Det <= 0 " << value << std::endl;
//
//            	std::cout << "OriginalPosition:" << std::endl;
//            	std::cout << e->geometry().corner(0) << std::endl;
//            	std::cout << e->geometry().corner(1) << std::endl;
//            	std::cout << e->geometry().corner(2) << std::endl;
//            	std::cout << e->geometry().corner(3) << std::endl << std::endl;
//
////            	std::cout << "NewPositions: " << std::endl;
////            	std::cout << e->geometry().corners(0)+  at_c<state_index>(x_.data).value(at_c<state_space_index>(e->geometry().corners(0)))<< std::endl;
////            	std::cout << e->geometry().corners(1) << std::endl;
////            	std::cout << e->geometry().corners(2) << std::endl;
////            	std::cout << e->geometry().corners(3) << std::endl << std::endl;
//
//            	std::cout << std::endl << std::endl;

           // }
//             for (int i = 0; i < dy_.M(); i++)
//                 for (int j = 0; j < dy_.N(); j++)
//                     if (dy_[i][j] == 0.0) std::cout << "Fail " << dy_[i][j] << std::endl; 
//               std::cout << "OuterDeterminante: " << temp.determinant() << std::endl;
//            std::cout << "Outertrace: " << temp[0][0] + temp[1][1] +temp[2][2] << std::endl;
            f_.update( id_ +at_c<state_index>(x_.data).derivative(at_c<state_space_index>(evaluators)) );
        }

        Scalar
        d0() const
        {

//              std::cout << "TestFunctionValue: " << f_() << std::endl;
//            return f_() - sp( g_, y_ );
        	return f_();
        }

        template<int row>
        Scalar d1_impl (const TestVarArg<row>& arg) const
        {
//             std::cout << "Ableitung:" << std::endl;
//            return f_.d1(arg.gradient) - sp( g_, arg.value );
        	return f_.d1(arg.gradient);
        }

        template<int row, int col>
        Scalar d2_impl (const TestVarArg<row>& arg1, const AnsatzVarArg<col>& arg2) const
        {
        	//  return f_.d2(arg1.derivative, arg2.derivative) +  0.5*f_.d1(arg1.derivative * dy_ * arg2.derivative + arg2.derivative * dy_ * arg1.derivative);
//             std::cout << "HesseMatrix: " << f_.d2(arg1.gradient+id_, arg2.gradient+id_) << std::endl;
          return f_.d2(arg1.gradient, arg2.gradient);
        }

    private:
        Integrand f_;
        const typename AnsatzVars::VariableSet& x_;
        typename AnsatzVars::Grid::template Codim<0>::Entity const* e;
        Dune::FieldVector<Scalar,AnsatzVars::template Components<state_index>::m> y_, g_;
        Dune::FieldMatrix<Scalar,AnsatzVars::template Components<state_index>::m,dim> id_, dy_;
        LinAlg::EuclideanScalarProduct sp;
    };
}

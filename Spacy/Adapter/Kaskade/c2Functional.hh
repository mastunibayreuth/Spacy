#pragma once

#include <memory>

#include "fem/assemble.hh"
#include "fem/istlinterface.hh"
#include "linalg/triplet.hh"

#include <Spacy/c1Operator.hh>
#include <Spacy/vector.hh>
#include <Spacy/vectorSpace.hh>
#include <Spacy/zeroVectorCreator.hh>
#include <Spacy/Util/Mixins/Eps.hh>
#include <Spacy/Util/Mixins/numberOfThreads.hh>
#include <Spacy/Util/Base/FunctionalBase.hh>

#include "directSolver.hh"
#include "vectorSpace.hh"
#include "vector.hh"
#include "operatorSpace.hh"
#include "linearOperator.hh"

namespace Spacy
{
    /** @addtogroup KaskadeGroup
     * @{
     */
    namespace Kaskade
    {
        /**
         * @brief %Functional interface for %Kaskade 7. Models a twice differentiable functional \f$f:X\rightarrow \mathbb{R}\f$.
         * @tparam FunctionalDefinition functional definition from %Kaskade 7
         * @see ::Spacy::C2Functional
         */
        template <class FunctionalDefinition>
        class C2Functional :
                public FunctionalBase,
                public Mixin::Eps,
                public Mixin::NumberOfThreads
        {
        public:
            /// %Kaskade::VariableSetDescription
            using VariableSetDescription = typename FunctionalDefinition::AnsatzVars;
            /// Coefficient vector type.
            using CoefficientVector = typename VariableSetDescription::template CoefficientVectorRepresentation<>::type;
            /// boost::fusion::vector<const Space0*,const Space1*,...>
            using Spaces = typename VariableSetDescription::Spaces;
            /// %Kaskade::VariationalFunctionalAssembler
            using Assembler = ::Kaskade::VariationalFunctionalAssembler< ::Kaskade::LinearizationAt<FunctionalDefinition> >;
            /// Matrix type
            using Matrix = ::Kaskade::MatrixAsTriplet<double>;
            /// operator for the description of the second derivative
            using KaskadeOperator = ::Kaskade::MatrixRepresentedOperator<Matrix,CoefficientVector,CoefficientVector>;

            using Linearization = LinearOperator<VariableSetDescription,VariableSetDescription>;

            /**
             * @brief Construct a twice differentiable functional \f$f: X\rightarrow \mathbb{R}\f$ from %Kaskade 7.
             * @param f operator definition from %Kaskade 7
             * @param domain domain space
             * @param rbegin first row to be considered in the definition of f
             * @param rend one after the last row to be considered in the definition of f
             * @param cbegin first column to be considered in the definition of f
             * @param cend one after the last column to be considered in the definition of f
             *
             * The optional parameters rbegin, rend, cbegin and cend can be used to define operators that correspond to parts of
             * a system of equation.
             */
            C2Functional(const FunctionalDefinition& f, const VectorSpace& domain,
                         int rbegin = 0, int rend = FunctionalDefinition::AnsatzVars::noOfVariables,
                         int cbegin = 0, int cend = FunctionalDefinition::TestVars::noOfVariables)
                : FunctionalBase(domain),
                  f_(f),
                  spaces_( extractSpaces<VariableSetDescription>(domain) ),
                  rhs_( zero(domain.dualSpace()) ),
                  rbegin_(rbegin), rend_(rend), cbegin_(cbegin), cend_(cend),
                  operatorSpace_( std::make_shared<VectorSpace>(
                                      LinearOperatorCreator<VariableSetDescription,VariableSetDescription>(domain,domain.dualSpace()) ,
                                      [](const ::Spacy::Vector& v)
            {
                using std::begin;
                using std::end;
                const auto& m = cast_ref<Linearization>(v).get();
                auto iend = end(m);
                auto result = 0.;
                for(auto iter = begin(m); iter!=iend; ++iter)
                    result += (*iter) * (*iter);
                return Real{sqrt(result)};
            } , true ) )
            {}


            /**
             * @brief Copy constructor.
             * @param g functional to copy from
             */
            C2Functional(const C2Functional& g)
                : FunctionalBase(g.domain()),
                  NumberOfThreads(g),
                  f_(g.f_), spaces_(g.spaces_),
                  A_(g.A_),
                  value_(g.value_),
                  old_X_f_(g.old_X_f_),
                  old_X_df_(g.old_X_df_),
                  old_X_ddf_(g.old_X_ddf_),
                  rhs_(g.rhs_),
                  onlyLowerTriangle_(g.onlyLowerTriangle_),
                  rbegin_(g.rbegin_), rend_(g.rend_), cbegin_(g.cbegin_), cend_(g.cend_),
                  solverCreator_(g.solverCreator_),
                  operatorSpace_(g.operatorSpace_),
                  normalComplianceParameter_(g.normalComplianceParameter_)
            {}

            /**
             * @brief Copy assignment.
             * @param g functional to copy from
             */
            C2Functional& operator=(const C2Functional& g)
            {
                setNumberOfThreads(g.getNumberOfThreads());
                f_ = g.f_;
                spaces_ = g.spaces_;
                A_ = g.A_;
                value_ = g.value_;
                old_X_f_ = g.old_X_f_;
                old_X_df_ = g.old_X_df_;
                old_X_ddf_ = g.old_X_ddf_;
                rhs_ = g.rhs_;
                onlyLowerTriangle_ = g.onlyLowerTriangle_;
                rbegin_ = g.rbegin_;
                rend_ = g.rend_;
                cbegin_ = g.cbegin_;
                cend_ = g.cend_;
                solverCreator_ = g.solverCreator_;
                operatorSpace_ = g.operatorSpace_;
                normalComplianceParameter_ = g.normalComplianceParameter_;
            }

            /**
             * @brief Move constructor.
             * @param g functional to move from
             */
            C2Functional(C2Functional&& g) = default;

            /**
             * @brief Move assignment.
             * @param g functional to move from
             */
            C2Functional& operator=(C2Functional&& g) = default;

            /**
             * @brief Apply functional.
             * @param x argument
             * @return \f$f(x)\f$
             */
            Real operator()(const ::Spacy::Vector& x) const
            {
                assembleFunctional(x);

                return value_;
            }

            // added by Stoecklein Matthias
            /**
             * @brief Get reference to the functional.
             * @return \f$&f\f$
             */
            auto & getFunctional()
            {
                return f_;
            }

            auto & getParameter()
            {
                return normalComplianceParameter_;
            }
            // added by Stoecklein Matthias
            /**
             * @brief Update path-parameter.
             */
            // not good delete  has to be done via getFunctional
            void updateParam(Real lambda)
            {
              //  std::cout << "UpdateGetsCalledAtAll: " << std::endl;
                f_.paramUpdate(get(lambda));
                normalComplianceParameter_ = get(lambda);
              //  std::cout << "NormalComplianceParameterUpdate: " << normalComplianceParameter_ << std::endl;

                // Does this work ??
                old_X_f_ = {};
                old_X_df_={};
                old_X_ddf_={};
                if(old_X_f_)
                    std::cout << "ResetFail: " << std::endl << std::endl;
            }




            /**
             * @brief Compute first directional derivative \f$f'(x) \in X^* \f$.
             *
             * @param x current iterate
             * @return \f$f'(x)\f$
             */
            ::Spacy::Vector d1(const ::Spacy::Vector& x) const
            {
                assembleGradient(x);

                return rhs_;
            }

            /**
             * @brief Compute second directional derivative \f$f''(x)dx\in X^* \f$.
             *
             * @param x current iterate
             * @param dx perturbation
             * @return \f$f''(x)dx\f$
             */
            ::Spacy::Vector d2(const ::Spacy::Vector& x, const ::Spacy::Vector& dx) const
            {
                assembleHessian(x);

                CoefficientVector dx_( VariableSetDescription::template CoefficientVectorRepresentation<>::init(spaces_) );
                copyToCoefficientVector<VariableSetDescription>(dx,dx_);
                CoefficientVector y_( VariableSetDescription::template CoefficientVectorRepresentation<>::init(spaces_) );

                A_.apply( dx_ , y_ );

                auto y = zero( domain().dualSpace() );
                copyFromCoefficientVector<VariableSetDescription>(y_,y);

                return y;
            }

            /**
             * @brief Access \f$f''(x)\f$ as linear operator \f$X\rightarrow X^*\f$.
             * @param x point of linearization
             * @see LinearOperator, ::Spacy::LinearOperator
             */
            auto hessian(const ::Spacy::Vector& x) const
            {
                assembleHessian(x);
                return Linearization{ A_ , *operatorSpace_ , solverCreator_ };
            }

            /// Access operator representing \f$f''\f$.
            const KaskadeOperator& A() const noexcept
            {
                return A_;
            }

            /// Access boost::fusion::vector of pointers to spaces.
            const Spaces& spaces() const noexcept
            {
                return spaces_;
            }

            /**
             * @brief Access onlyLowerTriangle flag.
             * @return true if only the lower triangle of a symmetric matrix is stored in the operator definition, else false
             */
            bool onlyLowerTriangle() const noexcept
            {
                return onlyLowerTriangle_;
            }

            /**
             * @brief Change solver creator.
             * @param f function/functor for the creation of a linear solver
             */
            void setSolverCreator(std::function<LinearSolver(const Linearization&)> f)
            {
                solverCreator_ = std::move(f);
            }

        private:
            /// Assemble \f$f(x)\f$.
            void assembleFunctional(const ::Spacy::Vector& x) const
            {
                if( old_X_f_ && old_X_f_ == x ) {

                    return;
                }

                VariableSetDescription variableSet(spaces_);
                typename VariableSetDescription::VariableSet u(variableSet);

                copy(x,u);

                Assembler assembler(spaces_);
                assembler.assemble(::Kaskade::linearization(f_,u) , Assembler::VALUE , getNumberOfThreads() );
                value_ = assembler.functional();

                assembler.assemble(::Kaskade::linearization(f_,u) , Assembler::RHS , getNumberOfThreads() );
                CoefficientVector coeff(assembler.rhs());
                boost::fusion::at_c<0>(coeff.data) = boost::fusion::at_c<0>(u.data).coefficients();
               // std::cout << "NormOfX: " << coeff.two_norm() << std::endl;

                old_X_f_ = x;
            }

            /// Assemble discrete representation of \f$f'(x)\f$.
            void assembleGradient(const ::Spacy::Vector& x) const
            {
                using boost::fusion::at_c;
                if( old_X_df_ && old_X_df_ == x )
                {
                 //   std::cout << "The same iterate: No new assembling of Gradient: " << std::endl;
                    return;
                }



                VariableSetDescription variableSet(spaces_);
                typename VariableSetDescription::VariableSet u(variableSet);

                copy(x,u);

                //std::cout << "BeginTestSensitivity: " << std::endl;
                //std::cout << "Parameter: "  << 2.0 * normalComplianceParameter_ << std::endl;
                //f_.paramUpdate(2.0*normalComplianceParameter_ );
                Assembler assembler(spaces_);
                //assembler.assemble(::Kaskade::linearization(f_,u) , Assembler::RHS , getNumberOfThreads() );
               // CoefficientVector coeff0(assembler.rhs());
                //std::cout <<  "NormRHS: " << coeff0.two_norm() << std::endl << std::endl;


                //std::cout << "Parameter: " << 0.5* normalComplianceParameter_ << std::endl;
                //f_.paramUpdate(0.5*normalComplianceParameter_);
               // assembler.assemble(::Kaskade::linearization(f_,u) , Assembler::RHS , getNumberOfThreads() );
                //CoefficientVector coeff1(assembler.rhs());
                //std::cout <<  "NormRHS: " << coeff1.two_norm() << std::endl << std::endl;
               // std::cout << "EndTestSensitiviy: " << std::endl << std::endl;




          //      f_.paramUpdate(normalComplianceParameter_);
              //  Assembler assembler(spaces_);
                assembler.assemble(::Kaskade::linearization(f_,u) , Assembler::RHS , getNumberOfThreads() );


              //  CoefficientVector coeff(assembler.rhs());
              //  std::cout << "Param: " << normalComplianceParameter_ << "   NormRHS: " << coeff.two_norm() << std::endl << std::endl;
               // boost::fusion::at_c<0>(coeff.data) = boost::fusion::at_c<0>(u.data).coefficients();
             //   std::cout << "StateNorm: " << coeff.two_norm() << std::endl << std::endl;


                copyFromCoefficientVector<VariableSetDescription>(CoefficientVector(assembler.rhs()),rhs_);

                old_X_df_ = x;
            }

            /// Assemble discrete representation of \f$f''(x)\f$.
            void assembleHessian(const ::Spacy::Vector& x) const
            {
                if( old_X_ddf_ && (old_X_ddf_==x) ) return;

                VariableSetDescription variableSet(spaces_);
                typename VariableSetDescription::VariableSet u(variableSet);

                copy(x,u);

                Assembler assembler(spaces_);
                assembler.assemble(::Kaskade::linearization(f_,u) , Assembler::MATRIX , getNumberOfThreads() );
                A_ = KaskadeOperator( assembler.template get<Matrix>(onlyLowerTriangle_,rbegin_,rend_,cbegin_,cend_) );

                old_X_ddf_ = x;
            }

            mutable FunctionalDefinition f_;
            Spaces spaces_;
            mutable KaskadeOperator A_ = {};
            mutable double value_ = 0;
            mutable ::Spacy::Vector old_X_f_{}, old_X_df_{}, old_X_ddf_{}, rhs_{};
            bool onlyLowerTriangle_ = false;
            int rbegin_=0, rend_=FunctionalDefinition::AnsatzVars::noOfVariables;
            int cbegin_=0, cend_=FunctionalDefinition::TestVars::noOfVariables;
            std::function<LinearSolver(const Linearization&)> solverCreator_ = [](const Linearization& f)
            {
                return makeDirectSolver<VariableSetDescription,VariableSetDescription>( f.A() ,
                                                                                        f.range() ,
                                                                                        f.domain() /*,
                                                                                                                                                             DirectType::MUMPS ,
                                                                                                                                                             MatrixProperties::GENERAL */);

            };
            std::shared_ptr<VectorSpace> operatorSpace_ = nullptr;
            double normalComplianceParameter_ = 0.0;
        };

        /**
         * @brief Convenient generation of a twice differentiable functional \f$f: X\rightarrow \mathbb{R}\f$ for %Kaskade 7.
         * @param f operator definition from %Kaskade 7
         * @param domain domain space
         * @param rbegin first row to be considered in the definition of f
         * @param rend one after the last row to be considered in the definition of f
         * @param cbegin first column to be considered in the definition of f
         * @param cend one after the last column to be considered in the definition of f
         * @return @ref C2Functional "::Spacy::Kaskade::C2Functional<FunctionalDefinition>( f, domain, rbegin, rend, cbegin, cend )"
         *
         * The optional parameters rbegin, rend, cbegin and cend can be used to define operators that correspond to parts of
         * a system of equation.
         */
        template <class FunctionalDefinition>
        auto makeC2Functional(const FunctionalDefinition& f, const VectorSpace& domain,
                              int rbegin = 0, int rend = FunctionalDefinition::AnsatzVars::noOfVariables,
                              int cbegin = 0, int cend = FunctionalDefinition::TestVars::noOfVariables)
        {
            return C2Functional<FunctionalDefinition>( f, domain , rbegin, rend , cbegin , cend );
        }
    }
    /** @} */
}

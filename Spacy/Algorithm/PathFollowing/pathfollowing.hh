// Copyright (C) 2017 by Stoecklein Matthias. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#pragma once

#include <memory>
#include <string>

#include "Spacy/c2Functional.hh"
#include "Spacy/operator.hh"
#include "Spacy/Util/Mixins/accuracy.hh"
#include "Spacy/Util/Mixins/Eps.hh"
#include "Spacy/Util/Mixins/iterativeRefinements.hh"
#include "Spacy/Util/Mixins/maxSteps.hh"
#include "Spacy/Util/Mixins/verbosity.hh"
#include <Spacy/Algorithm/dampingFactor.hh>
#include <Spacy/Util/Exceptions/callOfUndefinedFunctionException.hh>

namespace Spacy
{
/// @cond
class Vector;
/// @endcond

namespace PathFollowing
{



/**
 * @ingroup PathFollowingGroup
 * @brief Classical continuation method.
 *
 * This implements a classical continuation method to solve a parameter-dependent system \f$ F(x,\lambda) = 0 \f$, for scalar parameters.
 */
class ClassicalContinuation: public Mixin::AbsoluteAccuracy,
		public Mixin::RelativeAccuracy,
		public Mixin::Eps,
		public Mixin::Verbosity,
		public Mixin::MaxSteps
{
	enum class Result
	{
		Converged, Failed
	};

public:
        using InnerSolver = std::function<
            std::tuple<bool, Vector, Real, Real>(const Vector & x, const Real & lambda, const Real & theta)
        >;
	/**
	 * \brief Set up ClassicalContinuation method.
	 *
     * \param lambdaInit initial path parameter value
     * \param lambdaMax  maximal path parameter value
	 * \param initialStepSize  initial stepsize to update parameter \f$ \lambda \f$
	 * \param minStepSize  min stepsize to update parameter \f$ \lambda \f$
         * \param thetaMin
	 */
	ClassicalContinuation(const Real & lambdaInit, const Real & lambdaMax,
			const Real initialStepSize = 100.0, const Real minStepSize = 0.1,
                        const Real thetaMin = 1e-6);


	/// Set the solution function for the Newton system
	void setSolver(
			std::function<
					std::tuple<bool, Vector, Real, Real>(const Vector & x,
							const Real & lambda, const Real & theta)>);

	void setFirstStepSolver(
			std::function<
					std::tuple<bool, Vector>(const Vector & x,
							const Real & lambda)>);

	/// Set the plot function
	void setPlot(std::function<void(const Vector & x, unsigned int step)>);

    /**
     * \brief Solve problem via path-following method.
     * @param x0 initial guess
     */
    Vector solve(const Vector & x0) const;


private:

    /// Test feasibility of initial parameters and if the first solution on the path can be computed
   void testSetup(Real lambda, Real stepsize, Vector & x) const;

   /// Test if algorithm converged
   bool testResult(bool converged, int step, Real lambda, Real stepsize, const Vector & x) const;

   /// Test if algorithm converged
   Real updateStepSize(bool converged, int step, Real theta, Real thetaZero, Real stepsize_) const;


	/// Solve the Newton system for a fixed parameter
	std::function<
			std::tuple<bool, Vector, Real, Real>(const Vector & x,
                                                 const Real & lambda, const Real & theta)> innerSolver_ =[](const Vector & x,
            const Real & lambda, const Real & theta)
    {
        throw CallOfUndefinedFunctionException("Inner solver has not been set for path following");
        return std::make_tuple(false,x,lambda,theta);
    };

    /// Solve the first parameter system to get on the path
	std::function<
            std::tuple<bool, Vector>(const Vector & x, const Real & lambda)> solveFirstStep_ =
            [](const Vector & x, const Real & lambda)
    {
        throw CallOfUndefinedFunctionException("Inner solver for the first step has not been set for path following");
        return std::make_tuple(false,x);
    };

	/// Plot the solution \f$x(\lambda_{k})\f$ in each step
	std::function<void(const Vector & x, unsigned int step)> plot_ =
        [](const Vector & x, unsigned int step){ };

	mutable Result result_ = Result::Failed;

	// parameters for the classical continuation method
	const Real lambdaInit_;
	const Real lambdaMax_;

	const Real initialStepSize_;
	const Real minStepSize_;

	const Real thetaMin_;

    static constexpr double theta_ = 0.5;
    static constexpr double lowerBound_ = 0.414213562373095;  // sqrt(2)-1


};

//ClassicalContinuation::InnerSolver acrToInnerSolver(const ::Spacy::C2Functional& f,
//                                                    const ::Spacy::Vector & x,
//                                                    ::Spacy::Real lambda,
//                                                    const ::Spacy::Real & theta);
}
}


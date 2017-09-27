// Copyright (C) 2017 by Stoecklein Matthias. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#pragma once

#include <memory>
#include <string>

#include "Spacy/operator.hh"
#include "Spacy/Util/Mixins/accuracy.hh"
#include "Spacy/Util/Mixins/Eps.hh"
#include "Spacy/Util/Mixins/iterativeRefinements.hh"
#include "Spacy/Util/Mixins/maxSteps.hh"
#include "Spacy/Util/Mixins/verbosity.hh"
#include <Spacy/Algorithm/dampingFactor.hh>

namespace Spacy
{
/// @cond
class Vector;
/// @endcond

namespace PathFollowing
{


// ToDo Check numerical stabibility for small omega

/**
 * @ingroup PathFollowingGroup
 * @brief Classical continuation method.
 *
 * This implements a classical continuation method to sovle a parameter-dependent system \f$ F(x,\lambda) = 0 \f$, for scalar parameters.
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
	/**
	 * \brief Set up ClassicalContinuation method.
	 *
	 * \param lambdaInit initial parameter value
	 * \param lambdaMax  maximal parameter value
	 * \param initialStepSize  initial stepsize to update parameter \f$ \lambda \f$
	 * \param minStepSize  min stepsize to update parameter \f$ \lambda \f$
	 * \param omegaMin
	 */
	ClassicalContinuation(const Real & lambdaInit, const Real & lambdaMax,
			const Real initialStepSize = 100.0, const Real minStepSize = 0.1,
                        const Real thetaMin = 1e-6);

	/**
	 * \brief Solve problem via path-following method.
	 * @param x0 initial guess
	 */
	Vector solve(const Vector & x0) const;

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

private:
	/// Solve the Newton system for a fixed parameter
	std::function<
			std::tuple<bool, Vector, Real, Real>(const Vector & x,
					const Real & lambda, const Real & theta)> solveNewton_;

	/// Solve the first Newton system to get on the path
	std::function<
			std::tuple<bool, Vector>(const Vector & x, const Real & lambda)> solveFirstStep_;

	/// Plot the solution \f$x(\lambda_{k})\f$ in each step
	std::function<void(const Vector & x, unsigned int step)> plot_ =
	{ };

	mutable Result result_ = Result::Failed;

	// parameters for the classical continuation method
	const Real lambdaInit_;
	const Real lambdaMax_;

	const Real initialStepSize_;
	const Real minStepSize_;

	const Real thetaMin_;

        static constexpr double theta_ = 0.5;
	static constexpr double lowerBound_ = 0.414213562373095;  // sqrt(2)-1

	unsigned maxIter_;

};

}
}


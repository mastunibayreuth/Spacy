// Copyright (C) 2015 by Lars Lubkoll. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#include "pathfollowing.hh"

#include "Spacy/Util/Exceptions/notConvergedException.hh"
#include "Spacy/Util/Exceptions/invalidArgumentException.hh"
#include "Spacy/vector.hh"
#include <Spacy/Util/log.hh>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include <limits>

namespace Spacy
{
namespace PathFollowing
{

DEFINE_LOG_TAG( static const char* log_tag = "PathFollowing" );



ClassicalContinuation::ClassicalContinuation(const Real & lambdaInit,
		const Real & lambdaMax, const Real initialStepSize,
		const Real minStepSize, const Real thetaMin) :
		lambdaInit_(lambdaInit), lambdaMax_(lambdaMax), initialStepSize_(
				initialStepSize), minStepSize_(minStepSize), thetaMin_(thetaMin)
{

}

void ClassicalContinuation::setSolver(
		std::function<
				std::tuple<bool, Vector, Real, Real>(const Vector & x,
						const Real & lambda, const Real & theta)> f)
{
    innerSolver_ = std::move(f);
}

void ClassicalContinuation::setFirstStepSolver(
			std::function<
					std::tuple<bool, Vector>(const Vector & x,
							const Real & lambda)> f)
{
	solveFirstStep_ = std::move(f);
}


void ClassicalContinuation::setPlot(
		std::function<void(const Vector & x, unsigned int step)> f)
{
	plot_ = std::move(f);
}

Vector ClassicalContinuation::solve(const Vector & x0) const
{
	LOG_INFO(log_tag, "Starting iteration.")

	Real lambda = lambdaInit_;
	Real theta = 0.0;
	Real thetaZero = 0.25;

	auto x = x0;
	auto x_next = x;

        auto stepsize = initialStepSize_;

	bool converged = false;
	const int maxIt = getMaxSteps();

    testSetup(lambdaInit_, initialStepSize_, x);

    for (int step = 1; step <= maxIt; step++)
	{
             stepsize = std::min(stepsize, lambdaMax_ -lambda);
                LOG_SEPARATOR(log_tag);
                LOG(log_tag, "Iteration", step)
                 LOG(log_tag, "stepsize", stepsize, "lambda", lambda)
		LOG(log_tag, "|x|", norm(x))

		do
		{
            auto lambdaNext = lambda + stepsize;
                    std::cout << "LambdaTry: " << std::endl << std::endl;
                        std::tie(converged, x_next, theta, thetaZero) =
                                innerSolver_(x, lambdaNext, theta_);

            stepsize = updateStepSize(converged, step, theta,thetaZero, stepsize);

			if(converged)
			{
				x = x_next;
                                std::cout << "Converged:  For lambda:  " << lambdaNext << std::endl;
                                lambda = lambdaNext;
			}



        } while (!converged && stepsize > minStepSize_);

         converged = testResult(converged, step, lambda, stepsize,  x);

          LOG_SEPARATOR(log_tag);

          if(converged)
                 return x;

	}
    throw Exception::NotConverged("Max number of steps reached");
    return x;
}

void ClassicalContinuation::testSetup(Real lambda, Real stepsize, Vector & x) const
{

    if (lambda + stepsize > lambdaMax_ )
    {
        throw Exception::InvalidArgument("Initial stepsize for pathfollowing too large");
    }

    else if(stepsize < minStepSize_)
    {
       throw Exception::InvalidArgument("Initial stepsize smaller than minstepsize");
    }

    bool converged = false;
    Real theta = 0.0;
    Real thetaZero = 0.0;

   // std::tie(converged, x) = solveFirstStep_(x,lambda);

    std::tie(converged, x, theta, thetaZero) =
            innerSolver_(x, lambda, std::numeric_limits<double>::max());

    if(!converged)
    {
        throw Exception::NotConverged("Computation of initial iterate for path following");
    }

     plot_(x, 0);
}


bool ClassicalContinuation::testResult(bool converged, int step, Real lambda, Real stepsize, const Vector & x) const
{

    if (converged && lambda == lambdaMax_ )
    {
        plot_(x, step);
        result_ = Result::Converged;
        LOG(log_tag, "stepsize", stepsize, "lambda", lambda)
        LOG(log_tag, "|x|", norm(x))
        return true;
     }

    if (!converged)
      throw Exception::NotConverged("Computation of initial iterate for path following");



    else if(stepsize < minStepSize_)
    {
         throw Exception::NotConverged("Minimum stepsize without convergence reached");
    }

    return false;

}

Real ClassicalContinuation::updateStepSize(bool converged, int step, Real theta, Real thetaZero,Real stepsize_) const
{
    auto stepsize = stepsize_;
    if(converged)
    {
        std::cout << "Converged: For s = " << stepsize << std::endl;
        std::cout << "Theta: " << theta << std::endl << std::endl;
                        if(thetaZero <= thetaMin_)
                            stepsize *= lowerBound_/(2.0*thetaMin_);
                         else
                            stepsize *= lowerBound_/(2.0*thetaZero);
    }

    else
    {
        std::cout << "No Convergence for Theta: " << theta << std::endl;
        stepsize *= lowerBound_/(sqrt(1.0+4.0*theta)-1.0);
    }

    return stepsize;
}
//ClassicalContinuation::InnerSolver acr_to_inner_solver(const ::Spacy::C2Functional& f)
//{
//    return [&f](const Vector & x, Real lambda, const Real & theta)
//    {
//        f.updateParam(get(lambda));
//        Spacy::ACR::ACRSolver acr(f);
//        return acr.solveParam(x, lambda, theta );
//    };
//}
}
}

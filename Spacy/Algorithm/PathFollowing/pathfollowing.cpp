// Copyright (C) 2015 by Lars Lubkoll. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#include "pathfollowing.hh"

#include "Spacy/Util/Exceptions/singularOperatorException.hh"
#include "Spacy/vector.hh"
#include <Spacy/Util/log.hh>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>

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
	solveNewton_ = std::move(f);
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


	auto s = initialStepSize_;

	bool converged = false;
	const int maxIt = getMaxSteps();

	//std::tie(converged, x, theta, thetaZero) = solveNewton_(x, lambda, theta_);
	std::tie(converged, x) =  solveFirstStep_(x,lambda);
	plot_(x, 0);
	if (!converged)
	{
		std::cout << "No convergence for initial iterate x0 " << '\n';
		return x;
		//        		throw Exception::NotConverged("Newton"); //change
	}


	if (lambda + s > lambdaMax_)
	{
		std::cout << "Initial stepsize to large: " << '\n';
		return x;
	}

	for (int i = 1; i <= maxIt; i++)
	{
                LOG_SEPARATOR(log_tag);
                LOG(log_tag, "Iteration", i)
		LOG(log_tag, "stepsize", s, "lambda", lambda)
		LOG(log_tag, "|x|", norm(x))

                // leaves loop before check wether lambda max has been reached write in different function
		do
		{


			auto lambda_next = lambda + s;


			std::tie(converged, x_next, theta, thetaZero) = solveNewton_(x,
					lambda_next, theta_);



			// Kill one if condition
			if(converged)
			{
                                std::cout << "Converged: For s = " << s << std::endl;
				std::cout << "Theta: " << theta << std::endl;
                                //s *= lowerBound_/(sqrt(1.0+4.0*thetaZero)-1.0);
                                if(thetaZero <= thetaMin_)
                                    s *= lowerBound_/(2.0*thetaMin_);
                                 else
                                    s *= lowerBound_/(2.0*thetaZero);


			}

			else
			{
				std::cout << "Theta: " << theta << std::endl;
				s *= lowerBound_/(sqrt(1.0+4.0*theta)-1.0);
			}

//			if(theta > thetaMin_)
//				s *= (sqrt(1.0+4.0*theta_)-1.0)/(2.0*theta);
//
//			else
//   			s *= (sqrt(1.0+4.0*theta_)-1.0)/(2.0*thetaMin_);


//			s *= (lowerBound_)/(theta);

			if(converged)
			{
				x = x_next;
                                std::cout << "Converged:  For lambda:  " << lambda << std::endl;
				lambda = lambda_next;
                                std::cout << "  LambdaAfterUpdate " << lambda  << std::endl;
			}

			s = std::min(s, lambdaMax_ -lambda);
//			s = std::min(s, lambdaMax_ - lambda_next);

		} while (!converged && s > minStepSize_);


		if (!converged)
		{
			std::cout << "No convergence" << '\n';
			return x;
		}

		plot_(x, i);
                //Not well implemented
		if (lambda == lambdaMax_)
				{
					result_ = Result::Converged;
					std::cout << "Convergence " << '\n';

					LOG(log_tag, "stepsize", s, "lambda", lambda)
					LOG(log_tag, "|x|", norm(x))

					return x;
				}

		// not good
		 if (s < minStepSize_)
		{
			std::cout << "Stepsize too small" << '\n';
			return x;
		}


		
		 LOG_SEPARATOR(log_tag);
	}
	std::cout << "Reached max number of iterations " << '\n';
	return x;
}

}
}

// Copyright (C) 2015 by Lars Lubkoll. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#include "affineCovariantSolver.hh"

#include "Spacy/Algorithm/dampingFactor.hh"
#include "Spacy/Algorithm/CG/linearSolver.hh"
#include "Spacy/Algorithm/CG/triangularStateConstraintPreconditioner.hh"
#include "Spacy/Algorithm/CompositeStep/quadraticModel.hh"
#include "Spacy/Algorithm/Scalar/findGlobalMinimizer.hh"

#include "Spacy/inducedScalarProduct.hh"
#include "Spacy/Util/cast.hh"
#include "Spacy/Spaces/ProductSpace/vector.hh"
#include "Spacy/Util/Exceptions/regularityTestFailedException.hh"
#include "Spacy/Util/logger.hh"
#include <Spacy/zeroVectorCreator.hh>

#include <cmath>
#include <iostream>
#include <utility>

namespace Spacy
{
namespace
{
auto primalProjection(const Spacy::Vector& v)
{
	auto w = v;
	auto& w_ = cast_ref<ProductSpace::Vector>(w);
	w_.component(DUAL) *= 0;
	return w;
}

auto dualProjection(const Spacy::Vector& v)
{
	auto w = v;
	auto& w_ = cast_ref<ProductSpace::Vector>(w);
	w_.component(PRIMAL) *= 0;
	return w;
}

Logger<double> logNu("nu.log");
Logger<double> logTau("tau.log");
Logger<double> logOmegaC("omegaC.log");
Logger<double> logOmegaF("omegaf.log");
Logger<double> logThetaC("thetaC.log");
Logger<double> logEta("eta.log");
Logger<double> logDn("dn.log");
Logger<double> logDt("dt.log");
Logger<double> logDx("dx.log");
Logger<double> logDL("dL.log");
Logger<int> logRejected("rejected.log");
Logger<double> logCostFunctional("costFunctional.log");
Logger<bool> logConvexity("convex.log");
}

namespace CompositeStep
{
enum class AffineCovariantSolver::AcceptanceTest
{
	Passed, Failed, LeftAdmissibleDomain, TangentialStepFailed, NormalStepFailed
};

AffineCovariantSolver::AffineCovariantSolver(C2Functional N, C2Functional L,
		VectorSpace& domain,
		std::function<Vector(const Vector&, const Vector&)> retraction) :
		retraction_(retraction), dualUpdate_(linearRetraction), N_(
				std::move(N)), L_(std::move(L)), domain_(domain), chartSpace_(
				domain)
{
}

AffineCovariantSolver::AffineCovariantSolver(C2Functional N, C2Functional L,
		VectorSpace& totalSpace, VectorSpace& chartSpace,
		std::function<Vector(const Vector&, const Vector&)> retraction,
		std::function<Vector(const Vector&, const Vector&)> dualUpdate) :
		retraction_(retraction), dualUpdate_(dualUpdate), N_(std::move(N)), L_(
				std::move(L)), domain_(totalSpace), chartSpace_(chartSpace)
{
}

AffineCovariantSolver::AffineCovariantSolver(C2Functional N, C2Functional L,
		VectorSpace& domain) :
		retraction_(linearRetraction), dualUpdate_(linearRetraction), N_(
				std::move(N)), L_(std::move(L)), domain_(domain), chartSpace_(
				domain)
{
}

Vector AffineCovariantSolver::operator()()
{
	// New way of creating zeroVectors
	return operator()(zero(domain_));
}

Vector AffineCovariantSolver::operator()(const Vector& x0)
{
	std::cout << "StartIteration: " << std::endl;
	auto lastStepWasUndamped = false;
	auto x = x0;
//       logCostFunctional(L_(primalProjection(x)));
	logCostFunctional(get(L_(primalProjection(x))));

	std::cout << "starting composite step solver" << std::endl;

	converged_ = false;

	for (unsigned step = 1; step < getMaxSteps(); ++step)
	{

		normalStepMonitor = tangentialStepMonitor = StepMonitor::Accepted;

		// Is Constant for me !!
		domain_.setScalarProduct(PrimalInducedScalarProduct(N_.hessian(x)));
//        domain_.setScalarProduct( PrimalInducedScalarProduct( N_.hessian(primalProjection(x)) ) );

		if (verbose())
			std::cout << "\n ----------------- Composite Steps: Iteration "
					<< step << ". ----------------\n";
		if (verbose())
			std::cout << spacing << "**** Computing normal step **** "
					<< std::endl;
		auto Dn = computeNormalStep(x);
		auto norm_Dn = norm(Dn);
		DampingFactor nu = computeNormalStepDampingFactor(norm_Dn);

		if (getVerbosityLevel() > 1)
			std::cout << spacing2 << "|Dn|: " << norm(primalProjection(Dn))
					<< " nu = " << nu << std::endl;

		if (verbose())
			std::cout << spacing << "**** Computing lagrange multiplier ****"
					<< std::endl;
		x = updateLagrangeMultiplier(x);

		if (verbose())
			std::cout << spacing << "**** Computing tangential step ****"
					<< std::endl;
		auto Dt = computeTangentialStep(nu, x, Dn, lastStepWasUndamped);

		auto tau = DampingFactor
		{ 0. };
		Real norm_x = 0., norm_dx = 0.;
		auto ds = Dt;
		auto dx = Dt;

		if (verbose())
		{
			std::cout << spacing << "Tangential step length: " << norm(Dt)
					<< std::endl;
		}
		std::cout << "--- Inner Loop:begin --- " << std::endl;

		std::tie(tau, dx, ds, norm_x, norm_dx) = computeCompositeStep(nu,
				norm_Dn, x, Dn, Dt);

		std::cout << "--- Inner Loop:end --- " << std::endl;

		if (getContraction() < 0.25)
			x = retractPrimal(x, dx + ds);
		else
			x = retractPrimal(x, dx);
		logCostFunctional(get(L_(primalProjection(x))));
//         logCostFunctional(L_(primalProjection(x)));

		norm_x = norm(primalProjection(x));

		if (nu == 1 && tau == 1)
			lastStepWasUndamped = true;
		if (convergenceTest(nu, tau, norm_x, norm_dx))
		{
			converged_ =true;
			return x;
		}

		if (verbose())
			std::cout << spacing2 << "nu = " << nu << ", tau = " << tau
					<< ", |dx| = " << norm_dx << std::endl;
		if (verbose())
			std::cout << spacing2 << "|x| = " << norm_x << std::endl;
//        if( getVerbosityLevel() > 1 && norm_Dn > 0) std::cout << spacing2 << "(Dn,Dt) = " << Dn*Dt/(norm_Dn*norm(Dt)) << std::endl;
	} // end iteration

	return x;
}

std::tuple<bool, Vector, Real, Real> AffineCovariantSolver::solvePath(
		const Vector & x0, const Real & lambda, const Real & thetaGlobal)
{
	std::cout << "StartIteration: " << std::endl;
	std::cout << "OmegaC Initial: " << omegaC << std::endl;
//	omegaC = 1e-6;
//	omegaL =  Real(1e-16);
	auto lastStepWasUndamped = false;
	auto x = x0;

	Real theta = thetaGlobal;
	Real thetaNull = 2.0;

	Real norm_dx1 = 0.0;
	Real norm_dx2 = 0.0;

	bool converged = false;

	//       logCostFunctional(L_(primalProjection(x)));
	logCostFunctional(get(L_(primalProjection(x))));

	// update penalty Parameter

	// use std::function instead of
	std::cout << "starting composite step solver" << std::endl;
	for (unsigned step = 1; step < getMaxSteps(); ++step)
	{

		std::cout << "Step: " << step << " " << getMaxSteps() << '\n';

		normalStepMonitor = tangentialStepMonitor = StepMonitor::Accepted;

		domain_.setScalarProduct(PrimalInducedScalarProduct(N_.hessian(x)));
		//        domain_.setScalarProduct( PrimalInducedScalarProduct( N_.hessian(primalProjection(x)) ) );

		if (verbose())
			std::cout << "\n ----------------- Composite Steps: Iteration "
					<< step << ". ----------------\n";
		if (verbose())
			std::cout << spacing << "**** Computing normal step **** "
					<< std::endl;
		auto Dn = computeNormalStep(x);

		std::cout << "Normalstep computed : " << '\n';
		auto norm_Dn = norm(Dn);
		DampingFactor nu = computeNormalStepDampingFactor(norm_Dn);

		if (getVerbosityLevel() > 1)
			std::cout << spacing2 << "|Dn|: " << norm(primalProjection(Dn))
					<< " nu = " << nu << std::endl;

		if (verbose())
			std::cout << spacing << "**** Computing lagrange multiplier ****"
					<< std::endl;
		x = updateLagrangeMultiplier(x);

		if (verbose())
			std::cout << spacing << "**** Computing tangential step ****"
					<< std::endl;
		auto Dt = computeTangentialStep(nu, x, Dn, lastStepWasUndamped);

		auto tau = DampingFactor
		{ 0. };
		Real norm_x = 0., norm_dx = 0.;
		auto ds = Dt;
		auto dx = Dt;

		if (verbose())
		{
			std::cout << spacing << "Tangential step length: " << norm(Dt)
					<< std::endl;
		}
		std::cout << "--- Inner Loop:begin --- " << std::endl;

                std::tie(tau, dx, ds, norm_x, norm_dx) = computeCompositeStep(nu,
				norm_Dn, x, Dn, Dt);

		std::cout << "--- Inner Loop:end --- " << std::endl;

		if (getContraction() < 0.25)
			x = retractPrimal(x, dx + ds);
		else
			x = retractPrimal(x, dx);
		logCostFunctional(get(L_(primalProjection(x))));
		//         logCostFunctional(L_(primalProjection(x)));

		norm_x = norm(primalProjection(x));

		// Correct Norm what about lagrange multplier ???
		if (step > 2)
		{
			norm_dx1 = norm_dx2;
			norm_dx2 = norm_dx;

			if (norm_dx2 > 0.5 * norm_dx1)
			{
				theta = norm_dx2 / norm_dx1;
				std::cout << "No convergence for inner solver in Iterate: " << step << " " << lambda << " " << norm_dx1
						<< " " << norm_dx2 << '\n';
				// return make tupel
                                return std::make_tuple(converged, x, thetaNull, theta);
			}

//			else
//			{
//				if (std::isnan(get(theta)) || std::isinf(get(theta)))
//				{
//					std::cout << "Not a number: " << lambda << '\n';
//					theta = thetaGlobal ;
//				}
//			}
			norm_dx1 = norm_dx2;
		}

		// improve what happens when convergence in first step !!
		else if (step == 1)
		{
			norm_dx1 = norm_dx;
			theta = 0.15;
			thetaNull = theta;
		}

		else if (step == 2)
		{
			norm_dx2 = norm_dx;
			theta = norm_dx2 / norm_dx1;
			thetaNull = theta;

			/// change std::tie
			if (norm_dx2 > 0.25 * norm_dx1)
			{

				std::cout << "No convergence for inner solver first iterate: " << lambda << " " << norm_dx1
						<< " " << norm_dx2 << '\n';
                                return std::make_tuple(converged, x, thetaNull, theta);
			}
		}

//            if(step == 1) norm_dx1 = norm_dx;
//
//            if(step == 2) norm_dx2 = norm_dx;
//            {
//            	if(norm_dx2 > thetaGlobal * norm_dx1)
//            		{
//            			return std::tie(converged, x, theta);
//            			std::cout << "Not converged: " << lambda << std::endl;
//            		}
//
//            	else
//            	{
//
//            		theta = norm_dx2/norm_dx1;
//            		if(std::isnan(get(theta)))
//            			{
//            			std::cout << "Not a number: " << lambda << std::endl;
//            				theta = thetaGlobal/4.2;
//            			}
//            	}
//
//            }

		if (nu == 1 && tau == 1)
			lastStepWasUndamped = true;

		if (convergenceTest(nu, tau, norm_x, norm_dx))
		{
//			std::cout << "ComputeNormalStep Again: " << std::endl;
//			auto dn = computeNormalStep(x);
//			std::cout << "TestDNNorm: " << norm(dn) << std::endl;
//			domain_.setScalarProduct(PrimalInducedScalarProduct(N_.hessian(x)));
//			std::cout << "TestDNNorm: " << norm(dn) << std::endl;
			converged = true;
			std::cout << "Converged: " << lambda << std::endl;
                        return std::tie(converged, x, thetaNull, theta);
		}

		if (verbose())
			std::cout << spacing2 << "nu = " << nu << ", tau = " << tau
					<< ", |dx| = " << norm_dx << std::endl;
		if (verbose())
			std::cout << spacing2 << "|x| = " << norm_x << std::endl;
		//        if( getVerbosityLevel() > 1 && norm_Dn > 0) std::cout << spacing2 << "(Dn,Dt) = " << Dn*Dt/(norm_Dn*norm(Dt)) << std::endl;
	} // end iteration

        return std::make_tuple(converged, x, thetaNull, theta);
}

C2Functional & AffineCovariantSolver::getNormalFunctional()
{
	return N_;
}

C2Functional & AffineCovariantSolver::getTangentialFunctional()
{
	return L_;
}

void AffineCovariantSolver::setNormalFunctional(C2Functional N)
{
	N_ = std::move(N);
}

void AffineCovariantSolver::setTangentialFunctional( C2Functional L)
{
	L_ = std::move(L);
}

Vector AffineCovariantSolver::computeTangentialStep(DampingFactor nu,
		const Vector &x, const Vector& dn, bool lastStepWasUndamped) const
{
	if (!L_)
		return zero(chartSpace_);

	tangentialSolver = makeTangentialSolver(nu, x, lastStepWasUndamped);
	return primalProjection(
			tangentialSolver(
					primalProjection(-d1(L_, x))
							+ primalProjection(-nu * d2(L_, x)(dn))));
}

IndefiniteLinearSolver AffineCovariantSolver::makeTangentialSolver(
		DampingFactor nu, const Vector &x, bool lastStepWasUndamped) const
{
	Real trcgRelativeAccuracy = getMinimalAccuracy();
	if (nu == 1 && lastStepWasUndamped)
	{
		trcgRelativeAccuracy = max(getRelativeAccuracy(),
				min(getMinimalAccuracy(), omegaL * norm_dx_old));
		if (norm_dx_old > 0 && lastStepWasUndamped)
			trcgRelativeAccuracy = min(
					max(getRelativeAccuracy() / norm_dx_old,
							trcgRelativeAccuracy), getMinimalAccuracy());
		if (getVerbosityLevel() > 1)
		{
			std::cout << spacing2 << "relative accuracy = "
					<< trcgRelativeAccuracy << std::endl;
			std::cout << spacing2 << "absolute step length accuracy = "
					<< getRelativeAccuracy() * norm(x) << std::endl;
		}
	}

	auto setParams =
			[this,&x](auto& solver)
			{
				solver.setIterativeRefinements( getIterativeRefinements() );
				solver.setVerbosityLevel( getVerbosityLevel() );
				if( norm(primalProjection(x)) > 0)
				solver.setAbsoluteAccuracy( getRelativeAccuracy()*norm(primalProjection(x)) );
				else
				solver.setAbsoluteAccuracy( eps() );
				solver.setMaxSteps(getMaxSteps());
				solver.setRelativeAccuracy(0.01);
				//

			};

	//    std::unique_ptr<CGSolver> trcg = nullptr;

	if (is<CG::LinearSolver>(normalSolver))
	{
		const auto& cgSolver = cast_ref<CG::LinearSolver>(normalSolver);
		//       if( is<CG::TriangularStateConstraintPreconditioner>(cgSolver.P()))
		{
                    //ToDo  change in GitHub
//           auto trcg =  makeTCGSolver( L_.hessian(x) , cgSolver.P() ,
//                                        toDouble(trcgRelativeAccuracy) , eps() , verbose() );
			auto trcg = makeTCGSolver(L_.hessian(x), cgSolver.P(),
					trcgRelativeAccuracy, eps(), verbose());
			setParams(trcg);
			//Added by Stoecklein Matthias
			trcg.setRelativeAccuracy(1e-10);

			return IndefiniteLinearSolver(trcg);
		}
	}

	//    if( trcg == nullptr )
//        auto trcg = makeTCGSolver( L_.hessian(x) , normalSolver ,
//                                    toDouble(trcgRelativeAccuracy) , eps(), verbose() );
	auto trcg = makeTCGSolver(L_.hessian(x), normalSolver, trcgRelativeAccuracy,
			eps(), verbose());
	//  trcg.setIterativeRefinements(iterativeRefinements());
	//  trcg.setDetailedVerbosity(verbose_detailed());

	trcg.setRelativeAccuracy(0.1);
	if (norm(primalProjection(x)) > 0)
		trcg.setAbsoluteAccuracy(
				getRelativeAccuracy() * norm(primalProjection(x)));
	else
		trcg.setAbsoluteAccuracy(eps());



	// trcg.setMaxSteps(maxSteps());
	setParams(trcg);
	if (getVerbosityLevel() > 1)
	{
		std::cout << spacing2 << "relative accuracy = " << getRelativeAccuracy()
				<< std::endl;
		std::cout << spacing2 << "absolute step length accuracy = "
				<< getRelativeAccuracy() * norm(x) << std::endl;
	}

        // Added by Stoecklein Matthias
        trcg.setRelativeAccuracy(1e-6);
	return IndefiniteLinearSolver(trcg);
	//return std::move(trcg);
	//    return std::unique_ptr<IndefiniteLinearSolver>( trcg.release() );
}

Vector AffineCovariantSolver::computeNormalStep(const Vector &x) const
{
	if (!N_)
		return Vector(0 * x);

	tangentialSolver =
	{};
	normalSolver =
	{};
	normalSolver = N_.hessian(x) ^ -1;

	return computeMinimumNormCorrection(x);
}

Vector AffineCovariantSolver::computeSimplifiedNormalStep(
		const Vector &trial) const
{
	if (!N_)
		return zero(chartSpace_);
	return computeMinimumNormCorrection(trial);
}

Vector AffineCovariantSolver::computeMinimumNormCorrection(
		const Vector& x) const
{


	auto rhs = dualProjection(-d1(L_, x));


	//std::cout << "min norm correction rhs: " << norm(rhs) << std::endl;
	auto dn0 = zero(chartSpace_);
	if (is<CG::LinearSolver>(normalSolver))
	{
		std::cout << "CGSolver: " << '\n';
		auto& cgSolver = cast_ref<CG::LinearSolver>(normalSolver);
		cgSolver.set_eps(eps());
		cgSolver.setRelativeAccuracy(0.1);
		cgSolver.setVerbosity(verbose());
		cgSolver.setVerbosityLevel(getVerbosityLevel());

		cgSolver.setRelativeAccuracy(1e-10);
		//cgSolver.setIterativeRefinements(iterativeRefinements());
		//cgSolver.setMaxSteps(maxSteps());
		if (is<CG::TriangularStateConstraintPreconditioner>(cgSolver.P()))
		{
			const auto& P =
					cast_ref<CG::TriangularStateConstraintPreconditioner>(
							cgSolver.P());
			dn0 = P.kernelOffset(rhs);
			rhs -= cgSolver.A()(dn0);
		}

		else
		{
//			std::cout << "BeforePreconditionerCall: " << norm(dualProjection(rhs)) << std::endl;
//			std::cout << "RHSTransformation: " << '\n';
			dn0 = cgSolver.P()(rhs);
//			std::cout << "NormD0Test: " << norm(dn0) << std::endl;
			rhs -= cgSolver.A()(dn0);
			std::cout << "ComputeNormalStepRHSNorm: " << norm(rhs) << std::endl;
//			std::cout << "NormDualRHsTest: " << norm(dualProjection(rhs)) << std::endl;
		}
	}
	//     return dn0 + primalProjection( normalSolver( primalProjection(rhs) ) );
//	std::cout << "NormDualRHsTestBeforeCall: " << norm(dualProjection(rhs)) << std::endl;
	return dn0 + primalProjection(normalSolver(rhs));
}

Vector AffineCovariantSolver::updateLagrangeMultiplier(const Vector& x) const
{
	if (!N_ || !L_)
		return zero(chartSpace_);
	auto tmp = normalSolver(primalProjection(-d1(L_, x)));
	double normDL = get(norm(primalProjection(tmp)));
	logDL(normDL);
	std::cout << "Norm of projected gradient: " << normDL << std::endl;
	return dualUpdate_(x, dualProjection(tmp));
}
//      std::tuple<Real, Vector, Vector, Real, Real> AffineCovariantSolver::computeCompositeStep(Real & nu, Real norm_Dn,
//                                                         const Vector& x, const Vector& Dn, const Vector& Dt)
//     
std::tuple<DampingFactor, Vector, Vector, Real, Real> AffineCovariantSolver::computeCompositeStep(
		DampingFactor & nu, Real norm_Dn, const Vector& x, const Vector& Dn,
		const Vector& Dt)
{
	auto norm_Dt = norm(Dt);
	if (getVerbosityLevel() > 1)
		std::cout << spacing2 << "|Dt| = " << norm_Dt << " vs. "
				<< norm(primalProjection(Dt)) << std::endl;
	auto cubic = CompositeStep::makeCubicModel(nu, Dn, Dt, L_, x, omegaL);
	auto tau = computeTangentialStepDampingFactor(nu * norm_Dn, norm_Dt, cubic);

	auto ds = Vector
	{ 0 * Dt };
	auto dx = ds;
	auto eta = Real
	{ 1. };
	auto norm_x = norm(x);
	auto norm_dx = Real
	{ 0. };
	AcceptanceTest acceptanceTest = AcceptanceTest::Failed;

	int rej = -1;

	do
	{
		rej++;
		if (acceptanceTest == AcceptanceTest::LeftAdmissibleDomain)
			nu *= 0.5;
		else
			nu = computeNormalStepDampingFactor(norm_Dn);
		if (getVerbosityLevel() > 1)
			std::cout << spacing2 << "nu = " << nu << std::endl;

		auto quadraticModel = CompositeStep::makeQuadraticModel(nu, Dn, Dt, L_,
				x);
		auto cubicModel = CompositeStep::makeCubicModel(nu, Dn, Dt, L_, x,
				omegaL);

		if (acceptanceTest == AcceptanceTest::LeftAdmissibleDomain)
			tau *= 0.5;
		else
			tau = computeTangentialStepDampingFactor(nu * norm_Dn, norm_Dt,
					cubicModel);
		if (getVerbosityLevel() > 1)
			std::cout << spacing2 << "tau = " << tau << std::endl;
		auto q_tau = quadraticModel(get(tau));
		std::cout << spacing2 << "q(tau) = " << q_tau << std::endl;

		dx = nu * Dn + tau * Dt;
		norm_dx = norm(primalProjection(dx));
		if (getVerbosityLevel() > 1)
			std::cout << spacing2 << "|dx| = " << norm_dx << std::endl;
		auto trial = retractPrimal(x, dx);
//        std::cout << "|x| = " << norm(x) << ", |trial| = " << norm(trial) << ", norm(projected(trial)) = " << norm(primalProjection(trial)) << std::endl;

		if (!domain_.isAdmissible(trial))
			acceptanceTest = AcceptanceTest::LeftAdmissibleDomain;
		else
		{
                    //Changed by Stoecklein Matthias
                    if(acceptanceTest == AcceptanceTest::LeftAdmissibleDomain)
                    acceptanceTest = AcceptanceTest::Failed;
////
			if (verbose())
				std::cout << spacing
						<< "**** Computing simplified normal correction ****"
						<< std::endl;
			ds = computeSimplifiedNormalStep(trial);
//          auto trialplus = retractPrimal(x,dx+ds);
			ds += (nu - 1) * Dn;
//          std::cout << "ds1: " << norm(ds) << " vs. " << norm(primalProjection(ds)) << std::endl;
			auto trialplus = retractPrimal(x, dx + ds);
//          std::cout << "ds2: " << norm(ds) << std::endl;
//          std::cout << "soc: " << norm(trialplus) << std::endl;

			updateOmegaC(norm_x, norm_dx, norm(ds));
			eta = updateOmegaL(trialplus, q_tau, tau, norm_x, norm_dx,
					cubicModel);

			if (getVerbosityLevel() > 1)
				std::cout << spacing2 << "|ds| = " << norm(ds) << std::endl;

                        if (!domain_.isAdmissible(trialplus))
                        {
                            std::cout << "Simplified normal step left admissible domain:" << std::endl;
                            acceptanceTest = AcceptanceTest::LeftAdmissibleDomain;
                        }
		}

		regularityTest(nu, tau);

		if (acceptanceTest != AcceptanceTest::LeftAdmissibleDomain)
			acceptanceTest = acceptedSteps(norm_x, norm_dx, eta);

		if (acceptanceTest == AcceptanceTest::TangentialStepFailed
				&& omegaL
						< (1 + 0.25 * (1 - minimalDecrease()))
								* omegaL.previous())
		{
			if (getVerbosityLevel() > 1)
				std::cout << spacing2
						<< "Stagnating update of omegaL. Accepting Step."
						<< std::endl;
			acceptanceTest = AcceptanceTest::Passed;

			if (!acceptableRelaxedDecrease(eta))
			{
				if (getVerbosityLevel() > 1)
					std::cout << spacing2 << "Ignoring tangential step."
							<< std::endl;
				dx -= tau * Dt;
				trial = retractPrimal(x, dx);
				norm_dx = norm(dx);
			}
		}

		if (acceptanceTest == AcceptanceTest::Passed)
		{
			norm_dx_old = norm_dx;
		}

		if (getVerbosityLevel() > 1)
		{
			if (acceptanceTest == AcceptanceTest::Failed)
				std::cout << spacing2 << "Acceptance test failed." << std::endl;
			if (acceptanceTest == AcceptanceTest::NormalStepFailed)
				std::cout << spacing2 << "Acceptance test normal step failed."
						<< std::endl;
			if (acceptanceTest == AcceptanceTest::TangentialStepFailed)
				std::cout << spacing2
						<< "Acceptance test tangential step failed."
						<< std::endl;
			if (acceptanceTest == AcceptanceTest::LeftAdmissibleDomain)
				std::cout << spacing2
						<< "Acceptance test left admissible domain."
						<< std::endl;
			if (acceptanceTest == AcceptanceTest::Passed)
				std::cout << spacing2 << "Acceptance test passed." << std::endl;
			//~ if( normalStepMonitor == StepMonitor::Accepted) std::cout << spacing2 << "NormalStepMonitor::Accepted." << std::endl;
			//~ else std::cout << spacing2 << "NormalStepMonitor::Rejected" << std::endl;
			//~ if( tangentialStepMonitor == StepMonitor::Accepted) std::cout << spacing2 << "TangentialStepMonitor::Accepted." << std::endl;
			//~ else std::cout << spacing2 << "TangentialStepMonitor::Rejected" << std::endl;
		}
	} // end while (damping factors)
	while (acceptanceTest != AcceptanceTest::Passed);

	logNu(get(get(nu)));
	logTau(get(get(tau)));
	logOmegaC(get(get(omegaC)));
	logOmegaF(get(get(omegaL)));
	logDn(get(norm_Dn));
	logDt(get(norm_Dt));
	logDx(get(norm_dx));
	logThetaC(get(norm(ds) / norm_dx));
	logEta(get(eta));
	logRejected(rej);
	return std::make_tuple(tau, dx, ds, norm_x, norm_dx);
}

bool AffineCovariantSolver::convergenceTest(DampingFactor nu, DampingFactor tau,
		Real norm_x, Real norm_dx)
{
	logConvexity(tangentialSolver.isPositiveDefinite());

	//if( tangentialSolver && !tangentialSolver.isPositiveDefinite() ) return false;
	if (nu < 1 || tau < 0.95)
		return false;

	// TODO change that ??
	if (norm_dx < getRelativeAccuracy() * norm_x
			|| (norm_x < eps() && norm_dx < eps()))
	{

		std::cout << "OmegaC: " << omegaC << std::endl;
		// Added by Stoecklein Matthias
		std::cout << "getRelativeAccuracy() " << getRelativeAccuracy() << " "  << "eps()" << eps() <<  " " << "norm_dx" << norm_dx << '\n';
		if (verbose())
			std::cout << spacing << "Terminating (convergent)." << std::endl;
		return true;
	}

	return false;
}

void AffineCovariantSolver::updateOmegaC(Real norm_x, Real norm_dx,
		Real norm_ds)
{
	if (!N_)
		return;
	if (norm_dx < sqrt_eps() * norm_x
			&& getContraction() < getDesiredContraction())
		return;
	setContraction(norm_ds / norm_dx);
	//    if( getContraction() < 0.25 && ( norm_dx < sqrt_eps() * norm_x || norm_ds < eps() * norm_x ) ) return;

	if (!(normalStepMonitor == StepMonitor::Rejected
			&& tangentialStepMonitor == StepMonitor::Rejected)
			|| omegaC < 2 * getContraction() / norm_dx)
		{
		  omegaC = 2 * getContraction() / norm_dx;
		  std::cout << "OmegaCUpdate: " << omegaC << std::endl;
		}

	if (getVerbosityLevel() > 1)
		std::cout << spacing2 << "theta = " << getContraction() << ", omegaC: "
				<< omegaC << std::endl;
}

Real AffineCovariantSolver::updateOmegaL(const Vector& soc, Real q_tau,
		DampingFactor tau, Real norm_x, Real norm_dx,
		const CompositeStep::CubicModel& cubic)
{
	if (!tangentialSolver)
		return 1;

	Real eta = 1;
	if (abs(cubic(get(tau)) - cubic(0)) > sqrt_eps() * norm_x)
		eta = (L_(primalProjection(soc)) - cubic(0))
				/ (cubic(get(tau)) - cubic(0));
	else
		eta = 1;

	auto omegaLnew = (L_(primalProjection(soc)) - q_tau) * 6
			/ (norm_dx * norm_dx * norm_dx);

	if (!(abs(eta - 1) < 0.05 && omegaLnew > omegaL)
			&& (!(normalStepMonitor == StepMonitor::Rejected
					&& tangentialStepMonitor == StepMonitor::Rejected)
					|| omegaL < omegaLnew))
		omegaL = omegaLnew;

	if (getVerbosityLevel() > 1)
	{
		std::cout << spacing2 << "predicted decrease: "
				<< (cubic(get(tau)) - cubic(0));
//        std::cout << spacing2 << "cubic(tau): " << cubic(tau) << ", cubic(0): " << cubic(0) << std::endl;
//        std::cout << spacing2 << "L(primalProjection(soc)): " << L_(primalProjection(soc)) << ", |primalProjection(soc)| = " << norm(primalProjection(soc)) << std::endl;
		std::cout << spacing2 << "actual decrease: "
				<< (L_(primalProjection(soc)) - cubic(0)) << std::endl;
		std::cout << spacing2 << "omegaL: " << omegaL;
		std::cout << spacing2 << "eta: " << eta << std::endl;
	}

	return eta;
}
Vector AffineCovariantSolver::retractPrimal(const Vector& origin,
		const Vector& increment) const
{
	auto result = dualProjection(origin);
	result += primalProjection(
			retraction_(primalProjection(origin), primalProjection(increment)));
	return result;
}

DampingFactor AffineCovariantSolver::computeNormalStepDampingFactor(
		Real norm_Dn) const
{

	if (!N_)
		return DampingFactor(1);
	auto nu = DampingFactor(1);
	if (norm_Dn > eps() && abs(norm_Dn * omegaC) > eps())
		nu = min(1., getDesiredContraction() / (omegaC * norm_Dn));
	return nu;
}

DampingFactor AffineCovariantSolver::computeTangentialStepDampingFactor(
		Real norm_dn, Real norm_Dt,
		const CompositeStep::CubicModel& cubic) const
{
	if (!L_)
		return DampingFactor(1);
	if (norm_Dt < sqrt_eps())
		return DampingFactor(1);

	auto maxTau = Real
	{ 1. };
	if (pow(getRelaxedDesiredContraction() / omegaC, 2) - norm_dn * norm_dn > 0)
		maxTau = min(1.,
				sqrt(
						pow(2 * getRelaxedDesiredContraction() / omegaC, 2)
								- norm_dn * norm_dn) / norm_Dt);

	// Added by Stoecklein Matthias  Bug ??
	auto tau = DampingFactor(0.0);
//	if (maxTau >= 1e-12)
	if (maxTau >= 1e-8)
		tau = Scalar::findLogGlobalMinimizer(cubic, 1e-8, maxTau,
				getDampingAccuracy());
	if (tau <= 1e-8)
		tau = 0.0;
	return DampingFactor(tau);

//       auto tau = DampingFactor(1);
//             if( !L_ )
//                 return tau;
//             if( norm_Dt < sqrt_eps() )
//                 return tau;
// 
//             auto maxTau = Real{1.};
//             if( pow(getRelaxedDesiredContraction()/omegaC,2) - norm_dn*norm_dn > 0)
//                 maxTau = min( 1. , sqrt( pow( 2*getRelaxedDesiredContraction()/omegaC , 2 ) - norm_dn*norm_dn )/norm_Dt );
// 
//             return DampingFactor(Scalar::findGlobalMinimizer( cubic, 0, maxTau , getDampingAccuracy() ));

}

AffineCovariantSolver::AcceptanceTest AffineCovariantSolver::acceptedSteps(
		Real norm_x, Real norm_Dx, Real eta)
{
	if (norm_Dx < eps() * norm_x)
		return AcceptanceTest::Passed;

	if (!!L_ && !acceptableDecrease(eta))
	{
		if (getVerbosityLevel() > 1)
			std::cout << spacing2 << "Rejecting tangential step." << std::endl;
		tangentialStepMonitor = StepMonitor::Rejected;
		return AcceptanceTest::TangentialStepFailed;
	}

	if (!!N_ && !contractionIsAdmissible())
	{
		if (getVerbosityLevel() > 1)
			std::cout << spacing2 << "Rejecting normal step: "
					<< getContraction() << std::endl;
		normalStepMonitor = StepMonitor::Rejected;
		return AcceptanceTest::NormalStepFailed;
	}

	return AcceptanceTest::Passed;
}

void AffineCovariantSolver::regularityTest(DampingFactor nu,
		DampingFactor tau) const
{
	if (!regularityTestPassed(nu))
		throw RegularityTestFailedException(
				"AffineCovariantSolver::regularityTest (nu,...)", get(get(nu)));
//      if( !regularityTestPassed(tau) ) throw RegularityTestFailedException("AffineCovariantSolver::regularityTest (...,tau)",toDouble(tau));
}
}
}

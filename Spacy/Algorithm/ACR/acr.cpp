#include "acr.hh"

#include <Spacy/operator.hh>
#include <Spacy/Algorithm/dampingFactor.hh>
#include <Spacy/Algorithm/CG/linearSolver.hh>
#include <Spacy/Algorithm/CompositeStep/quadraticModel.hh>
#include <Spacy/Algorithm/CG/cg.hh>
#include <Spacy/Algorithm/Scalar/findGlobalMinimizer.hh>
#include <Spacy/Algorithm/CG/linearSolver.hh>
#include <Spacy/zeroVectorCreator.hh>
#include <Spacy/Util/cast.hh>
#include <Spacy/Util/log.hh>
#include "Spacy/Util/Exceptions/notConvergedException.hh"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <utility>



namespace Spacy
{
namespace ACR
{
DEFINE_LOG_TAG(static const char* log_tag = "ACR");

CompositeStep::CubicModel makeCubicModel(const Vector& dx,
		const C2Functional& f, const Vector& x, Spacy::Real omega)
{
	return CompositeStep::CubicModel(
			CompositeStep::makeQuadraticModel(Spacy::DampingFactor(0.0), dx, dx,
					f, x),
			CompositeStep::makeQuadraticNormModel(Spacy::DampingFactor(0.0), dx,
					dx), omega);
}

ACRSolver::ACRSolver(C2Functional f, double eta1, double eta2, double epsilon,
		double relativeAccuracy, double omegaMax, double lambdaMax) :
		Mixin::RelativeAccuracy(relativeAccuracy), f_(std::move(f)), domain_(
				f_.domain()), eta1_(eta1), eta2_(eta2), epsilon_(epsilon), omegaMax_(
				omegaMax), lambdaMin_(lambdaMax)
{
}

Vector ACRSolver::operator()()
{
	return operator()(zero(domain_));
}

Vector ACRSolver::operator()(const Vector& x0)
{
	LOG_INFO(log_tag, "Starting ACR-Solver.")

	auto x = x0;
	auto dx = x;
	Real dxQNorm = 0.0;
	Real lambda = 1.0;

	for (unsigned step = 1; step <= getMaxSteps(); ++step)
	{

                LOG_SEPARATOR(log_tag);
		LOG(log_tag, "Iteration", step)

		stepMonitor = StepMonitor::Accepted;
		// TODO domain_.setScalarProduct( );

		std::tie(dx,dxQNorm) = computeStep(x);
		std::cout << "DxQNorm: " << dxQNorm << std::endl;

                if (dxQNorm < epsilon_)
		{
			std::cout << "Converged: " << std::endl;
			std::cout << "f_.d1(x)*f_.d1(x): " <<  f_.d1(x)*f_.d1(x) << std::endl;
			return x;
		}

		std::cout << "Test wether dx is a direction of descent :" << f_(x + dx*1e-9) -f_(x)
									<< std::endl;
		std::cout << "f_.d1(x)*f_.d1(x): " <<  f_.d1(x)*f_.d1(x) << std::endl;
		std::cout << std::setprecision(20) << "NormDxBeforeUpdate: " << norm(dx)
				<< std::endl;


		do
		{

			auto cubicModel = makeCubicModel(dx, f_, x, omega_);
			LOG_INFO(log_tag, "Computing damping factor")


			lambda = Scalar::findLogGlobalMinimizer(cubicModel, 1e-12, 1e5, 0.1);

			std::cout << "LambdaComputed: " << lambda << std::endl;

			auto dxDummy = get(lambda) * dx;

			std::cout << "DirectFunctionValueBeforeUpdate: f_(x+dx) :" << f_(x + dxDummy)
							<< std::endl;

			nonlinUpdate_(x, dxDummy);

			std::cout << "DirectFunctionValueAfterUpdate: f_(x+dx) :" << f_(x + dxDummy)
										<< std::endl;

			LOG(log_tag, "lambda: ", lambda, " omega: ", omega_, "|dx|: " , norm(dxDummy), " cubicModel: ", cubicModel(lambda))

			if ((stepMonitor = acceptanceTest(x, dxDummy, lambda, cubicModel))
					== StepMonitor::Accepted)
			{
				x += dxDummy;
				output_(x);
			}

			else
				LOG_INFO(log_tag, "Rejected")

				// Modifikation von omega
				omega_ = weightChange(omega_);


		}
		while (stepMonitor == StepMonitor::Rejected && omega_ <= omegaMax_ );
		if(stepMonitor == StepMonitor::Rejected  )
		{
			 std::cout << "Not Converged: " << '\n';
			 return x;
		}
	  LOG_SEPARATOR(log_tag);
}
  throw Exception::NotConverged("Maximum number of iterations reached");
return x;
}


std::tuple<bool, Vector, Real, Real> ACRSolver::solveParam(
                const Vector & x0,  Real lambda_, Real thetaGlobal)
{
        std::cout << "StartIteration: " << '\n';

        auto x = x0;
        auto dx = x;

        Real theta = 0.1;
        Real thetaZero = 0.1;

        Real normDx1 = 0.0;
        Real normDx2 = 0.0;

        Real normDx = 0.0;
        Real lambda = 1.0;

        Real dxQNorm = 0.0;

        bool contraction = false;

        for (unsigned step = 1; step <= getMaxSteps(); ++step)
        {

           LOG_SEPARATOR(log_tag);
           LOG(log_tag, "Iteration", step)

           stepMonitor = StepMonitor::Accepted;
           // TODO domain_.setScalarProduct( );

           std::tie(dx,dxQNorm) = computeStep(x);

           normDx = sqrt(dxQNorm);

           std::cout << std::setprecision(20) << "DxQNorm: " << normDx << std::endl;

           std::tie(contraction,normDx1,normDx2,thetaZero,theta) = checkContraction(step,normDx, normDx1, normDx2, thetaZero, thetaGlobal);

           if(!contraction)
               return std::make_tuple(contraction,x,thetaZero,theta);

           do
           {
                   auto cubicModel = makeCubicModel(dx, f_, x, omega_);
                   LOG_INFO(log_tag, "Computing damping factor")

                   lambda = Scalar::findLogGlobalMinimizer(cubicModel, 1e-12, 1e2, 0.1);

                   std::cout << "LambdaComputed: " << lambda << std::endl;

                   auto dxDummy = get(lambda) * dx;

                   nonlinUpdate_(x, dxDummy);


                   LOG(log_tag, "lambda: ", lambda, " omega: ", omega_, "|dx|: " , norm(dxDummy), " cubicModel: ", cubicModel(lambda))

                   if ((stepMonitor = acceptanceTest(x, dxDummy, lambda, cubicModel))
                                   == StepMonitor::Accepted)
                   {
                           x += dxDummy;
                           output_(x);
                   }

                   else
                           LOG_INFO(log_tag, "Rejected")

                           // Modifikation von omega
                           omega_ = weightChange(omega_);


           }
           while (stepMonitor == StepMonitor::Rejected && omega_ <= omegaMax_ );   // Check if every case is covered !!

           if(stepMonitor == StepMonitor::Rejected)
           {
               std::cout << "This should not happen: " << '\n';
                return std::make_tuple(false,x,1.0, 1.0);
           }


           if(convergenceTest(dxQNorm,x))
               return std::make_tuple(true,x, thetaZero, theta);


           LOG_SEPARATOR(log_tag);


}

        throw Exception::NotConverged("Maximum number of iterations reached");
        return std::make_tuple(false, x, thetaZero, theta);
}




void ACRSolver::setNonlinUpdate(
	std::function<void(const Vector & x, Vector& dx)> nonlinUpdate)
{
nonlinUpdate_ = std::move(nonlinUpdate);
}

void ACRSolver::setOutput(std::function<void(const Vector& dx)> outputUpdate)
{
output_ = std::move(outputUpdate);
}

C2Functional &  ACRSolver::getFunctional()
{
    return f_;
}



ACRSolver::StepMonitor ACRSolver::acceptanceTest(const Vector &x,
	const Vector &dx, const Real & lambda,
	const CompositeStep::CubicModel& cubicModel)
{

// Prevent dividing by zero e.g. in the case x_0 = opimal solution
// Consider the Case numerator = nominator = 0

const auto diffModel = cubicModel(lambda) - cubicModel(0.0);
const auto diffFunction = (f_(x + dx) - f_(x));

//std::cout << "m(dx)-m(0): " << diffModel << std::endl;
//std::cout << "f_(x+dx)-f_(x): " << diffFunction << std::endl;

if ( !std::signbit(get(diffFunction)) ||  std::isnan(get(diffFunction))  )
{
	rho_ = -1.0;
}


else {
//Change that Condition
rho_ = diffFunction / diffModel;
if (std::isnan(get(rho_)))
	rho_ = -1.0;

}
LOG(log_tag, "f_(x+dx): ", f_(x+dx), "f_(x): ", f_(x), "CubicModel(lambda): ", cubicModel(lambda), "CubicModel(0): ", cubicModel(0), "rho: ", rho_, "eta1 ", eta1_, "eta2 ", eta2_ )

if (rho_ >= eta1_)
	return StepMonitor::Accepted;

return StepMonitor::Rejected;
}

Spacy::Real ACRSolver::weightChange(Spacy::Real omega) const
{

LOG(log_tag, "rho: ", rho_, "eta1 ", eta1_, "eta2 ", eta2_ )

if (rho_ > eta2_)
	omega *= 0.5;
else if (rho_ < eta1_)
	omega *= 2.0;

return omega;
}

std::tuple<Vector,Real> ACRSolver::computeStep(const Spacy::Vector &x) const
{

	// Why not  get solverCreator directly from f ????
	// make this a member
LinearSolver preconditioner =
{ };
preconditioner = f_.hessian(x) ^ -1;


auto & cgSolver = cast_ref<CG::LinearSolver>(preconditioner);
auto tcg = makeTCGSolver(f_.hessian(x), cgSolver.P());


//auto tcg = makeTRCGSolver(f_.hessian(x), cgSolver.P());
//
    //       auto tcg =  makeTCGSolver( f_.hessian(x) , f_.hessian(x).solver());
//tcg.setRelativeAccuracy(getRelativeAccuracy());
// Added by Stoecklein Matthias
tcg.setRelativeAccuracy(1e-10);

return tcg.solve(-f_.d1(x));
}


std::tuple<bool,Real,Real,Real,Real> ACRSolver::checkContraction(int step, Real normDx,Real normDx1_, Real normDx2_, Real thetaZero_, Real thetaGlobal) const
{
    Real normDx1 = normDx1_;
    Real normDx2 = normDx2_;
    Real theta = thetaZero_;
    Real thetaZero = thetaZero_;
    bool contracted = true;

    if (step > 2)
    {
            normDx1 = normDx2;
            normDx2 = normDx;

              theta = normDx2 / normDx1;

              if (std::isnan(get(theta)) || std::isinf(get(theta)))
                  throw Exception::InvalidArgument("Contraction Factor is NaN");

            if (theta > thetaGlobal )
            {
                  std::cout << "No convergence for inner solver in Iterate: " << step  << " "<< theta << " " << normDx1
                                    << " " << normDx2 << '\n';
                  contracted = false;
            }
    }

   else if(step ==1)
        normDx1 = normDx;

    else if (step == 2)
    {
            normDx2 = normDx;
            theta = normDx2 / normDx1;
            thetaZero = theta;

            if (std::isnan(get(theta)) || std::isinf(get(theta)))
                throw Exception::InvalidArgument("Contraction Factor is NaN");


          if (theta > thetaGlobal )
          {
                std::cout << "No convergence for inner solver in Iterate: " << step  << " " << theta <<" " << normDx1
                                  << " " << normDx2 << '\n';
                contracted = false;
          }
    }

    return std::make_tuple(contracted, normDx1, normDx2, thetaZero, theta);

}


 bool ACRSolver::convergenceTest( Real dxQNorm, const Vector & x) const
 {

          if (dxQNorm < epsilon_)
          {
                  std::cout << "ACR Converged: " << '\n';
                  std::cout << "f_.d1(x)*f_.d1(x): " <<  f_.d1(x)*f_.d1(x) << '\n';
                  return true;
 }

          else
              return false;
}
}
}

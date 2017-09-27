/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*  This file is part of the library KASKADE 7                               */
/*    see http://www.zib.de/Numerik/numsoft/kaskade7/                        */
/*                                                                           */
/*  Copyright (C) 2002-2013 Zuse Institute Berlin                            */
/*                                                                           */
/*  KASKADE 7 is distributed under the terms of the ZIB Academic License.    */
/*    see $KASKADE/academic.txt                                              */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define FUSION_MAX_VECTOR_SIZE 15
#define SPACY_ENABLE_LOGGING
#include <chrono>
#include <iostream>

#include <dune/grid/config.h>
#include <dune/grid/uggrid.hh>

#include <Spacy/Adapter/kaskade.hh>
#include <Spacy/Algorithm/CompositeStep/affineCovariantSolver.hh>

#include "fem/gridmanager.hh"
#include "fem/lagrangespace.hh"
#include <fem/boundaryspace.hh>
#include "fem/forEach.hh"
#include "io/vtk.hh"
#include "utilities/kaskopt.hh"
#include "utilities/gridGeneration.hh"
#include "fem/constantspace.hh"

#include <Spacy/Algorithm/CG/linearSolver.hh>
// #include <Spacy/Algorithm/CG/trivialPreconditioner.hh>
// #include <Spacy/Algorithm/CG/directPreconditioner.hh>
#include <Spacy/Adapter/Kaskade/directBlockPreconditioner.hh>
#include <Spacy/Algorithm/PathFollowing/pathfollowing.hh>

#include "linalg/simpleLAPmatrix.hh"

#include "parameters.hh"
#include  "setup.hh"

#define NCOMPONENTS 1
#include "nonlinear_elasticity.hh"
// #include "functional.hh"
// #include "trackingFunctional.hh"

#include <mg/apcg.hh>
#include <mg/pcg.hh>
#include <mg/multigrid.hh>
#include <mg/additiveMultigrid.hh>

#include <boost/type_index.hpp>
#include <fung/examples/rubber/mooney_rivlin.hh>
//#include <dune/grid/utility/hierarchicsearch.hh>
#include <algorithm>
#include <fem/forEach.hh>
#include <linalg/apcg.hh>
#include <fem/particle.hh>

int main(int argc, char *argv[])
{
	/// Change displacements to deformations in nonlinear_elasticity.hh

	using namespace Kaskade;

	constexpr int dim = 3;

	const std::string parFile("nonlinear_elasticity.parset");

	ProblemParameters parameters = readParameters(parFile);

	const unsigned refinements = parameters.refinements;
	const unsigned order = parameters.order;
	const bool onlyLowerTriangle = parameters.onlyLowerTriangle;
	const unsigned numberOfCubes = parameters.numberOfCubes;
	const double obstacle = parameters.obstacle;
	const double regularization = parameters.regularization;

	const double initial_boundary_force = parameters.initial_boundary_force;
	const double reference_boundary_force = parameters.reference_boundary_force;

	const double referenceNormalComplianceParameter =
			parameters.referenceNormalComplianceParameter;
	const double initialNormalComplianceParameter =
			parameters.initialNormalComplianceParameter;
	const double maxNormalComplianceParameter =
			parameters.maxNormalComplianceParameter;

	const double desiredAccuracy = parameters.desiredAccuracy;
	const double eps = parameters.eps;
	const double alpha = parameters.alpha;

	const int maxSteps = parameters.maxSteps;
	const int iterativeRefinements = parameters.iterativeRefinements;
	const int FEorder = parameters.FEorder;
	const int verbose = parameters.verbose;

	const double desContr = parameters.desContr;
	const double relDesContr = parameters.relDesContr;
	const double maxContr = parameters.maxContr;

	static constexpr int stateId = 0;
	static constexpr int controlId = 1;
	static constexpr int adjointId = 2;

	const double x1 = 0.0;
	const double x2= 0.0;
	const double x3 = 0.0;
	const double lx1 = 2.0;
	const double lx2 = 2.0;
	const double lx3 = 0.2;


	using Scalar = double;
	using Grid = Dune::UGGrid<dim>;

	using Vector = Dune::FieldVector<Scalar, dim>;

	const Vector reference_force_density
	{ 0., 0., reference_boundary_force };



	// Creates grid by connecting cubes
	//auto grid = createGridFactory<Grid>(numberOfCubes, dim);

//	auto grid = createCuboidFactory<Grid>(x1, x2, x3, lx1, lx2, lx3);
	auto grid = createDefaultCuboidFactory<Grid>();
	GridManager<Grid> gridManager(std::move(grid));
	gridManager.globalRefine(refinements);



	using H1SpaceRef = FEFunctionSpace<ContinuousLagrangeMapper<double,typename Grid::LeafGridView> >;
	using SpacesRef = boost::fusion::vector<H1SpaceRef const*>;
	using RefVariableDescriptions = boost::fusion::vector< Variable<SpaceIndex<0>,Components<dim>,VariableId<0> > >;
	using RefVariableSetDescription = VariableSetDescription<SpacesRef,RefVariableDescriptions>;
	using VariableSetRef = typename RefVariableSetDescription::VariableSet;
	using CoefficientVector = typename RefVariableSetDescription::template CoefficientVectorRepresentation<>::type;

	H1SpaceRef deformationSpaceRef(gridManager,
			gridManager.grid().leafGridView(), order);

	SpacesRef spacesRef(&deformationSpaceRef);
	RefVariableSetDescription descriptionRef(spacesRef,
	{ "y" });

    std::cout << "Degrees of freedom state: " << descriptionRef.degreesOfFreedom() << std::endl;
	VariableSetRef y(descriptionRef);

	// Boundary Mapper has not been implemented yet for discontinuous boundary functions
	using L2Space = FEFunctionSpace<BoundaryMapper<ContinuousLagrangeMapper, double,typename Grid::LeafGridView> >;
	using ControlSpace = boost::fusion::vector<L2Space const*>;
	using ControlVariableDescriptions = boost::fusion::vector< Variable<SpaceIndex<0>,Components<dim>,VariableId<0> > >;
	using ControlVariableSetDescription = VariableSetDescription<ControlSpace,ControlVariableDescriptions>;
	using ControlVariableSet = typename ControlVariableSetDescription::VariableSet;

	// numberOf BoundarySegements has to be computed  because respective grid function does not work
	std::vector<int> totalIndexSet(0);
	std::vector<int> partialIndexSetId =
	{ 1 };

	{
		const auto & gv = gridManager.grid().leafGridView();

		std::size_t numberOfBoundaryFaces = 0;

		forEachBoundaryFace(gv, [&numberOfBoundaryFaces](const auto & face)
		{	numberOfBoundaryFaces++;});
		totalIndexSet.resize(numberOfBoundaryFaces, 0);

		// write more compact with lambda
		const Vector up
		{ 0., 0., 1.0 };

		forEachBoundaryFace(gv, [&up, &totalIndexSet] (const auto & face)
		{
			if(face.centerUnitOuterNormal()*up > 0.5)
			{
				const auto boundarySegmentId = face.boundarySegmentIndex();
				totalIndexSet.at(face.boundarySegmentIndex()) = 1;
			}
		});
	}

	L2Space l2Space(gridManager, gridManager.grid().leafGridView(), order,
			totalIndexSet, partialIndexSetId);

	ControlSpace controlSpace(&l2Space);
	ControlVariableSetDescription controlDescription(controlSpace,
	{ "u" });


    std::cout << "Degrees of Freedom Control: " << controlDescription.degreesOfFreedom() << std::endl;
	ControlVariableSet u(controlDescription);
	u = 0.0;

	int counter = 1;

	// initialize boundary force
	std::for_each(std::begin(boost::fusion::at_c<0>(u.data).coefficients()),
			std::end(boost::fusion::at_c<0>(u.data).coefficients()),
			[dim, &counter, &reference_force_density] (auto & entry)
			{
//				if(counter > 20)
//				{
//					entry[dim-1] = -100.0-(0.05*counter);
//				}

				entry = reference_force_density;
				counter++;

			});

	std::cout << "USize:"
			<< boost::fusion::at_c<0>(u.data).coefficients().size() << '\n';

// Creating material function
	using Matrix = Dune::FieldMatrix<double,dim,dim>;

	auto y00 = FunG::LinearAlgebra::unitMatrix<Matrix>();

	//auto refIntegrand  = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
			//		Matrix>(0.08625, 0.08625, 0.68875, -1.895, y00);

	//auto refIntegrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
		//	Matrix>(100, 100, 1, -1, y00);

	 auto refIntegrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>,
		FunG::LN,Matrix>(3.690000000,  0.410000000,   2.090000000,  -13.20000000,
		y00);

	using RefFunctional = NonlinearElasticityFunctionalDistributed<decltype(refIntegrand),RefVariableSetDescription,ControlVariableSet>;
	using RefAssembler = VariationalFunctionalAssembler<LinearizationAt<RefFunctional> >;
	using RefOperator = AssembledGalerkinOperator<RefAssembler>;

	auto FRef = RefFunctional(refIntegrand, referenceNormalComplianceParameter,
			u, obstacle);

	RefAssembler assemblerRef(gridManager, spacesRef);

	assemblerRef.assemble(linearization(FRef, y));

	//  Spacy

	auto domainRef =
			Spacy::Kaskade::makeHilbertSpace<RefVariableSetDescription>(
					spacesRef,
					{ 0u });
	auto f = Spacy::Kaskade::makeC2Functional(FRef, domainRef);

	using FTypeRef = Spacy::Kaskade::C2Functional<RefFunctional>;
	using FLinRef = typename Spacy::Kaskade::C2Functional<RefFunctional>::Linearization;

	using X0 = Dune::BlockVector<Dune::FieldVector<double,dim>>;
	using NMatrix = NumaBCRSMatrix<Dune::FieldMatrix<double,dim,dim>>;

	NMatrix Amat(assemblerRef.template get<0, 0>(), false);

	std::cout << "Nonzeroes: " << Amat.nonzeroes() << " " << Amat.N() << '\n';

//	using boost::typeindex::type_id_with_cvr;
//	std::cout << "TypeABlockOuter: "
//			<< type_id_with_cvr<decltype(assemblerRef.template get<0, 0>())>().pretty_name()
//			<< '\n';

	// Does not work ? Matrix does not get updated ???
	auto mg = makeBPX(Amat, gridManager);

/// Get Matrix for mg from f !!!
	std::function<Spacy::LinearSolver(const FLinRef&)> precond =
			[&mg,&gridManager](const FLinRef& f)
			{
		                const auto & dummy = f.get();
						auto bcrs = dummy.template toBCRS<3>();
					//	auto NumAMatrix_ = NumaBCRSMatrix<Dune::FieldMatrix<double, dim, dim>>(*bcrs,true);
						// More Efficient  Create Matrix Outside once and update it
						// avoid Copy somehow  check wether NumAMatrix_ lives   ho
						mg = makeBPX( NumaBCRSMatrix<Dune::FieldMatrix<double, dim, dim>>(*bcrs,true), gridManager);

				return Spacy::makeTCGSolver(f,Spacy::Kaskade::makePreconditioner<typename FTypeRef::VariableSetDescription,typename FTypeRef::VariableSetDescription>(f.range(),f.domain(),mg));
			};

	// compute solution by ACR

    std::for_each(std::begin(boost::fusion::at_c<0>(u.data).coefficients()),
            std::end(boost::fusion::at_c<0>(u.data).coefficients()),
            [dim, &counter, &reference_force_density] (auto & entry)
            {
//				if(counter > 20)
//				{
//					entry[dim-1] = -100.0-(0.05*counter);
//				}

                entry = reference_force_density;
                counter++;

            });

    VariableSetRef reference_deformation(descriptionRef);


    auto coeffIter = boost::fusion::at_c<0>(reference_deformation.data).coefficients().begin();
    auto coeffEnd = boost::fusion::at_c<0>(reference_deformation.data).coefficients().end();
    auto last = gridManager.grid().leafGridView().template end<3>();

    //std::cout << "ParticleSize: " << particles_.size() << '\n';

//    reference_deformation = 0.0;
//    for (auto it = gridManager.grid().leafGridView().template begin<3>(); it != last;
//            ++(it),++(coeffIter))
//    {
//            if( (it->geometry().center())[0] <= 1.50001 && (it->geometry().center())[0] >= 0.49999   && (it->geometry().center())[1] <= 1.50001 && (it->geometry().center())[1] >= 0.49999   )
//                if(!( ( (it->geometry().center())[0] >= 0.7500  &&  (it->geometry().center())[0] <= 1.250000 )   && ( (it->geometry().center())[1] >= 0.7500  &&  (it->geometry().center())[1]<= 1.25000 )))
//                   {
//                    (coeffIter->operator[](2)) -= 0.4;
//                    (coeffIter->operator[](2)) *= 1.5;
//     }

//    }

    f.setSolverCreator(precond);
    Spacy::ACR::ACRSolver solver(f);
    auto resultRef = solver();

    Spacy::Kaskade::copy(resultRef, reference_deformation);

	IoOptions optionsRef;
	optionsRef.outputType = IoOptions::ascii;

	std::string outfilenameRef("ReferenceDeformation");
	writeVTKFile(gridManager.grid().leafGridView(), reference_deformation,
			outfilenameRef, optionsRef, 1);



	using H1Space = FEFunctionSpace<ContinuousLagrangeMapper<double,Grid::LeafGridView> >;
	using Spaces = boost::fusion::vector<H1Space const*,L2Space const *, H1Space const *>;

	using VariableDescriptions = boost::fusion::vector<Variable<SpaceIndex<0>,Components<3>,VariableId<stateId> >,
	Variable<SpaceIndex<1>,Components<3>,VariableId<controlId> > ,
	Variable<SpaceIndex<2>,Components<3>,VariableId<adjointId> > >;

	using VariableSetDescription = VariableSetDescription<Spaces,VariableDescriptions>;
	using VariableSet = VariableSetDescription::VariableSet;

	// Create Spaces
	H1Space deformationSpace(gridManager, gridManager.grid().leafGridView(),
			order);
	H1Space adjointState(gridManager, gridManager.grid().leafGridView(), order);

	Spaces spaces(&deformationSpace, &l2Space, &adjointState);
	std::string names[3] =
	{ "y", "u", "p" };

	VariableSetDescription description(spaces, names);
	VariableSet x(description);

	auto domain = Spacy::Kaskade::makeHilbertSpace<VariableSetDescription>(
			spaces,
			{ 0u, 1u },
			{ 2u });
	std::cout << "Created Domain" << std::endl;


    domain.setRestriction([&FRef,&y, &x,&assemblerRef,&gridManager] (const Spacy::Vector & x_) {

            static unsigned int counter = 0;

            IoOptions options;
            options.outputType = IoOptions::ascii;

            Spacy::Kaskade::copy( x_, x);
            ParticleCloud<dim, Grid,
                    decltype(gridManager.grid().leafGridView().indexSet())> cloud(
                    gridManager.grid());

               cloud.moveLinear(boost::fusion::at_c<0>(x.data));


               cloud.updateGlobalDisplacements(x);

               const auto & particles = cloud.getParticles();

               auto par = x;
               par = 0;
               auto count = 0;

               std::for_each(std::begin(boost::fusion::at_c<0>(par.data).coefficients()),
                       std::end(boost::fusion::at_c<0>(par.data).coefficients()),
                       [particles,count] (auto & entry) mutable
                       {

                           if((particles[count].orientationChanged_))
                           {
                               entry = 1.0;
                           }

                           count++;

                       });

               std::string name = "OrientationChange" + std::to_string(counter);

   //            writeVTKFile(gridManager.grid().leafGridView(), par, name, options,
   //                    1);

               name = "Deformation_" + std::to_string(counter);

   //            writeVTKFile(gridManager.grid().leafGridView(), x, name, options,
   //                    1);




        Spacy::Kaskade::copy( x_, x);

        //Check wether dimension is correct !!
        boost::fusion::at_c<0>(y.data) = boost::fusion::at_c<0>(x.data);
        assemblerRef.assemble(::Kaskade::linearization(FRef,y) , Assembler::VALUE  );
        auto fvalue = assemblerRef.functional();
        std::cout << "FValue: " << fvalue << std::endl;

                   if(std::isnan(fvalue) || std::isinf(fvalue))
                    {
                       std::cout << "EvenGetChanges: " << std::endl;
                        std::cout << "FValue: Is: " << fvalue << "  Reduce step length" << '\n';
                        return false;
           }
                    return true;
               }


    );








	auto y0 = FunG::LinearAlgebra::unitMatrix<Matrix>();

//   // Normal step functional with direct solver
	//auto integrand  = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
				//		Matrix>(0.08625, 0.08625, 0.68875, -1.895, y0);
	auto integrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>,
			FunG::LN,Matrix>(3.690000000,  0.410000000,   2.090000000,  -13.20000000,
			y0);
//	auto integrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
//			Matrix>(100, 100, 1, -1, y0);





    auto penaltyFunctional = Spacy::Kaskade::makeC2Functional(
            VolumePenaltyFunctional<decltype(integrand), stateId, controlId,adjointId, double, VariableSetDescription>(integrand), domain);

/*
     domain.setRestriction([&penaltyFunctional] (const Spacy::Vector & x) {
         auto value = penaltyFunctional(x);
         if(std::isnan(get(value)) || std::isinf(get(value)))
         {
             std::cout << "FValue: Is inf: Reduce step length" << '\n';
             return false;
}
         return true;
     }
     );*/

	auto fn = Spacy::Kaskade::makeC2Functional(
			NormalStepFunctional<decltype(integrand),
					decltype(reference_deformation), stateId, controlId,
					adjointId, double, VariableSetDescription>(regularization,
					reference_deformation, integrand,
					initialNormalComplianceParameter, obstacle), domain);

	using NSF_type = NormalStepFunctional<decltype(integrand), decltype(reference_deformation), stateId,controlId,adjointId,double,VariableSetDescription>;
	using FN_type = Spacy::Kaskade::C2Functional<NSF_type>;
	using FNLin = FN_type::Linearization;

//		std::function<Spacy::LinearSolver(const FNLin&)> myPCG2SolverCreator =
//				[&description, &sharedSparsityPattern, &gridManager](const FNLin& f)
//				{
//					static constexpr int dimension = dim;
//					Spacy::Kaskade::DirectBlockPreconditioner<NSF_type> P(f.A(),description,f.domain(),f.range());
//					return Spacy::makeTCGSolver( f , P);
//				};

	////

	// Consider that again
	const auto sharedSparsityPattern = Amat.getPattern();

	std::function<Spacy::LinearSolver(const FNLin&)> myPCG2SolverCreator =
			[&description, &sharedSparsityPattern, &gridManager](const FNLin& f)
			{
				static constexpr int dimension = dim;
				Spacy::Kaskade::DirectBlockPreconditioner<NSF_type/*,decltype(mg), decltype(Amat), decltype(gridManager)*/ /*,decltype(sharedSparsityPattern),dimension,decltype(gridManager)*/> P(f.A(),description,f.domain(),f.range()/*,mg, Amat, gridManager,sharedSparsityPattern,gridManager*/);
				return Spacy::makeTCGSolver( f , P);
			};

   fn.setSolverCreator(myPCG2SolverCreator);

//	using NormalStepFunctional = NormalStepFunctional<decltype(integrand),
//	decltype(reference_deformation), stateId, controlId,
//	adjointId, double, VariableSetDescription>;
//
//	using Assembler = VariationalFunctionalAssembler<LinearizationAt<NormalStepFunctional> >;
//	using Operator = AssembledGalerkinOperator<Assembler>;
//	auto norm = NormalStepFunctional(regularization, reference_deformation,
//			integrand, initialNormalComplianceParameter);
//
//	Assembler assembler(gridManager, spaces);
//	assembler.assemble(linearization(norm, x));
//	Operator A(assembler);
//	auto trip = A.getTriplet();
//	std::cout << "Symmetric " << trip.isSymmetric() << '\n';

	auto ft = Spacy::Kaskade::makeC2Functional(
			TangentialStepFunctional<decltype(integrand),
					decltype(reference_deformation), stateId, controlId,
					adjointId, double, VariableSetDescription>(regularization,
					reference_deformation, integrand,
					initialNormalComplianceParameter, obstacle), domain);
    auto testFunctional = Spacy::Kaskade::makeC2Functional(
        TestFunctional<decltype(integrand),
                    decltype(reference_deformation), stateId, controlId,
                    adjointId, double, VariableSetDescription>(regularization,
                                                               reference_deformation, integrand,
                                                               initialNormalComplianceParameter, obstacle), domain);







	std::cout << "set up solver" << std::endl;
	// algorithm and parameters
	auto cs = Spacy::CompositeStep::AffineCovariantSolver(fn, ft, domain);
	cs.setRelativeAccuracy(desiredAccuracy);
	cs.set_eps(eps);
	cs.setVerbosityLevel(2);
	cs.setMaxSteps(maxSteps);
	cs.setIterativeRefinements(iterativeRefinements);
	cs.setDesiredContraction(desContr);
	cs.setRelaxedDesiredContraction(relDesContr);
	cs.setMaximalContraction(maxContr);







	auto path = Spacy::PathFollowing::ClassicalContinuation(
			initialNormalComplianceParameter, maxNormalComplianceParameter);

	std::function<
			std::tuple<bool, ::Spacy::Vector, ::Spacy::Real, ::Spacy::Real>(
					const ::Spacy::Vector & x, ::Spacy::Real lambda,
					const ::Spacy::Real & theta)> solveFunction =
			[desiredAccuracy, eps, maxSteps, iterativeRefinements, desContr, relDesContr, maxContr, &fn, &ft, &domain, &cs, &testFunctional ] (const ::Spacy::Vector & x, ::Spacy::Real lambda, const ::Spacy::Real & theta) mutable
			{


                fn.updateParam(get(lambda));


//				auto c = Spacy::CompositeStep::AffineCovariantSolver(fn, ft, domain);
//					c.setRelativeAccuracy(desiredAccuracy);
//					c.setEps(eps);
//					c.setVerbosityLevel(2);
//					c.setMaxSteps(maxSteps);
//					c.setIterativeRefinements(iterativeRefinements);
//					c.setDesiredContraction(desContr);
//					c.setRelaxedDesiredContraction(relDesContr);
//					c.setMaximalContraction(maxContr);
//
//					return c.solvePath(x, lambda, theta );

                auto & tang = cs.getTangentialFunctional();
                auto & norm = cs.getNormalFunctional();
                auto tPointer = tang.template target<decltype(ft)>();
                auto nPointer = norm.template target<decltype(fn)>();
      //          	using boost::typeindex::type_id_with_cvr;
	//std::cout << "TangentialFunctionalType  : "
	//		<< type_id_with_cvr<decltype(pointer)>().pretty_name()
	//		<< '\n';

                tPointer->updateParam(get(lambda));
                nPointer->updateParam(get(lambda));
				ft.updateParam(get(lambda));

                // *tPointer = ft;
			//	cs.setNormalFunctional(fn);
		//		cs.setTangentialFunctional(ft);
             //  std::cout << "LambdaBeforeSolve: " << lambda << std::endl;
				return cs.solvePath(x, lambda, theta );
			};

	std::function<
				std::tuple<bool, ::Spacy::Vector> (
						const ::Spacy::Vector & x, ::Spacy::Real lambda) >solveFirstStep =
				[&fn, &ft, &cs] (const ::Spacy::Vector & x, ::Spacy::Real lambda) mutable
				{
					// Necessary because ft is casted to Spacy::C2Functional

        //std::cout << "LambdaTest: " << lambda << std::endl;
					//fn.updateParam(get(lambda));

				//	ft.updateParam(get(lambda));
				//	cs.setNormalFunctional(fn);
				//	cs.setTangentialFunctional(ft);


                    auto & tang = cs.getTangentialFunctional();
                    auto & norm = cs.getNormalFunctional();
                    auto tPointer = tang.template target<decltype(ft)>();
                    auto nPointer = norm.template target<decltype(fn)>();
                    //          	using boost::typeindex::type_id_with_cvr;
                    //std::cout << "TangentialFunctionalType  : "
                    //		<< type_id_with_cvr<decltype(pointer)>().pretty_name()
                    //		<< '\n';

                    tPointer->updateParam(get(lambda));
                    nPointer->updateParam(get(lambda));

					return std::make_tuple(cs.converged_,cs());
				};

	std::function<void(const ::Spacy::Vector & solution, unsigned int step)> plotFunction =
			[&gridManager,&x, order ] (const ::Spacy::Vector & solution, unsigned int step)
			{
				Spacy::Kaskade::copy(solution, x);
				IoOptions options;
				options.outputType = IoOptions::ascii;
				std::string outfilename = "Deformation_" + std::to_string(step);

				writeVTKFile(gridManager.grid().leafGridView(), x, outfilename, options,
                        order);
			};

	path.setSolver(solveFunction);
	path.setFirstStepSolver(solveFirstStep);
	path.setPlot(plotFunction);

	const auto result = path.solve(Spacy::zero(domain));

	IoOptions options;
	options.outputType = IoOptions::ascii;
	std::string outfilename = "Deformation_u";


    //ToDo check if solution is minimizer !
    // Create Second Project for non constraint Optimization
    // Change penalty to x^4
    // Build check wether fvalue is infinity
    // increase cg accuracy
    // Check sensibility with respect to alpha

    // Check wether arg.derivative has the correct dimension
    // Bug Composite Step  norm ds < norm dx < 0.25 ??
    // Check wether ds

	writeVTKFile(gridManager.grid().leafGridView(), u, outfilename, options,
			order);


}


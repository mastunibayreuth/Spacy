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

#include <boost/type_index.hpp>
#include <fung/examples/rubber/mooney_rivlin.hh>

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

	using Scalar = double;
	using Grid = Dune::UGGrid<dim>;

	using Vector = Dune::FieldVector<Scalar, dim>;

	const Vector reference_force_density
	{ 0., 0., reference_boundary_force };

	// Creates grid by connecting cubes
	auto grid = createGridFactory<Grid>(numberOfCubes, dim);

	GridManager<Grid> gridManager(std::move(grid));
	gridManager.globalRefine(refinements);

	const int init_size = gridManager.grid().size(dim) * dim;

	std::vector<double> init(init_size, 0.0);

	for (int i = 2; i < init_size; i += 3)
	{
		init[i] = -1.5;
	}
	using Scalar0 = double;
	using H1Space0 = FEFunctionSpace<ContinuousLagrangeMapper<double,typename Grid::LeafGridView> >;
	using Spaces0 = boost::fusion::vector<H1Space0 const*>;
	using VariableDescriptions0 = boost::fusion::vector< Variable<SpaceIndex<0>,Components<dim>,VariableId<0> > >;
	using VariableSetDescription0 = VariableSetDescription<Spaces0,VariableDescriptions0>;
	using VariableSet0 = typename VariableSetDescription0::VariableSet;

	H1Space0 deformationSpace0(gridManager, gridManager.grid().leafGridView(),
			order);

	Spaces0 spaces0(&deformationSpace0);
	VariableSetDescription0 description0(spaces0,
	{ "y" });

	using Matrix0 = Dune::FieldMatrix<double,dim,dim>;

	auto y00 = FunG::LinearAlgebra::unitMatrix<Matrix0>();
	auto integrand0 = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
			Matrix0>(100, 100, 1, -1, y00);

	using Functional0 = NonlinearElasticityFunctional<decltype(integrand0),VariableSetDescription0>;
	using Assembler0 = VariationalFunctionalAssembler<LinearizationAt<Functional0> >;

	auto F0 = Functional0(integrand0, reference_force_density,
			referenceNormalComplianceParameter);
	Assembler0 assembler0(gridManager, spaces0);

	VariableSet0 y(description0);

	/*   VariableSet0 reference_deformation(description0);
	 reference_deformation.read(init.begin());*/

	assembler0.assemble(linearization(F0, y));
	//  Spacy

	auto domain0 = Spacy::Kaskade::makeHilbertSpace<VariableSetDescription0>(
			spaces0,
			{ 0u });
	auto f = Spacy::Kaskade::makeC2Functional(F0, domain0);

	using FType0 = Spacy::Kaskade::C2Functional<Functional0>;
	using FLin0 = typename Spacy::Kaskade::C2Functional<Functional0>::Linearization;

	using X0 = Dune::BlockVector<Dune::FieldVector<double,dim>>;
	using NMatrix = NumaBCRSMatrix<Dune::FieldMatrix<double,dim,dim>>;

	NMatrix Amat(assembler0.template get<0, 0>(), true);

	auto mg = makeBPX(Amat, gridManager);

	std::function<Spacy::LinearSolver(const FLin0&)> precond =
			[&mg](const FLin0& f)
			{
				return Spacy::makeTCGSolver(f,Spacy::Kaskade::makePreconditioner<typename FType0::VariableSetDescription,typename FType0::VariableSetDescription>(f.range(),f.domain(),mg));
			};

	// compute solution by ACR

	f.setSolverCreator(precond);
	Spacy::ACR::ACRSolver solver(f);
	auto result0 = solver();

	VariableSet0 reference_deformation(description0);
	Spacy::Kaskade::copy(result0, reference_deformation);

	IoOptions options0;
	options0.outputType = IoOptions::ascii;

	std::string outfilename0("ReferenceDeformation");
	writeVTKFile(gridManager.grid().leafGridView(), reference_deformation,
			outfilename0, options0, 1);

	// Is GridManager Changed ??
	// Get ReferenceDeformation
	//auto reference_deformation = setupFactory<dim, ProblemParameters, Grid,  ::Kaskade::GridManager<Grid>,  Vector>(parameters, gridManager, reference_force_density );

	// Define Spaces
	using H1Space = FEFunctionSpace<ContinuousLagrangeMapper<double,Grid::LeafGridView> >;
	using ConstantControlSpace = FEFunctionSpace<ConstantMapper<double,Grid::LeafGridView> >;
	using Spaces = boost::fusion::vector<H1Space const*,ConstantControlSpace const *, H1Space const *>;

	using VariableDescriptions = boost::fusion::vector<Variable<SpaceIndex<0>,Components<3>,VariableId<stateId> >,
	Variable<SpaceIndex<1>,Components<3>,VariableId<controlId> > ,
	Variable<SpaceIndex<2>,Components<3>,VariableId<adjointId> > >;

	using VariableSetDescription = VariableSetDescription<Spaces,VariableDescriptions>;
	using VariableSet = VariableSetDescription::VariableSet;

	// Create Spaces
	H1Space deformationSpace(gridManager, gridManager.grid().leafGridView(),
			order);
	H1Space adjointState(gridManager, gridManager.grid().leafGridView(), order);

	ConstantControlSpace controlSpace(gridManager,
			gridManager.grid().leafGridView(), 0);

	Spaces spaces(&deformationSpace, &controlSpace, &adjointState);
	std::string names[3] =
	{ "y", "u", "p" };

	VariableSetDescription description(spaces, names);
	VariableSet x(description);

	// Control not threedimensional
	/*    std::cout << "Degrees of Freedom Control: " << description.degreesOfFreedom(1,2) << std::endl;*/

	using Matrix = Dune::FieldMatrix<double,dim,dim>;

	auto y0 = FunG::LinearAlgebra::unitMatrix<Matrix>();
	auto integrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
			Matrix>(100, 100, 1, -1, y0);

	using Functional = NonlinearElasticityFunctionalControl<decltype(integrand),VariableSetDescription>;
//	using ParamFunctional = ParameterFunctional<decltype(integrand),VariableSetDescription>;

	auto F = Functional(integrand);
//	auto ParamF = ParamFunctional(integrand);

	auto domain = Spacy::Kaskade::makeHilbertSpace<VariableSetDescription>(
			spaces,
			{ 0u, 1u },
			{ 2u });
	std::cout << "Created Domain" << std::endl;

	/*	auto c1 = Spacy::Kaskade::makeC1Operator(ParamF, domain);
	 c1.setSolverCreator(precond);*/

//   // Normal step functional with direct solver
	auto integrand2 = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
			Matrix>(100, 100, 1, -1, y0);

	double normalComplianceParameter = initialNormalComplianceParameter;

	auto fn = Spacy::Kaskade::makeC2Functional(
			NormalStepFunctional<decltype(integrand2),
					decltype(reference_deformation), stateId, controlId,
					adjointId, double, VariableSetDescription>(regularization,
					reference_deformation, integrand2,
					normalComplianceParameter), domain);

	using NSF_type = NormalStepFunctional<decltype(integrand), decltype(reference_deformation), stateId,controlId,adjointId,double,VariableSetDescription>;
	using FN_type = Spacy::Kaskade::C2Functional<NSF_type>;
	using FNLin = FN_type::Linearization;

	std::function<Spacy::LinearSolver(const FNLin&)> myPCG2SolverCreator =
			[&description](const FNLin& f)
			{
				Spacy::Kaskade::DirectBlockPreconditioner<NSF_type> P(f.A(),description,f.domain(),f.range());
				return Spacy::makeTCGSolver( f , P);
			};

//	fn.setSolverCreator(myPCG2SolverCreator);

	auto ft = Spacy::Kaskade::makeC2Functional(
			TangentialStepFunctional<decltype(integrand2),
					decltype(reference_deformation), stateId, controlId,
					adjointId, double, VariableSetDescription>(regularization,
					reference_deformation, integrand2,
					normalComplianceParameter), domain);

	std::cout << "set up solver" << std::endl;
	// algorithm and parameters
	/*auto cs = Spacy::CompositeStep::AffineCovariantSolver(fn, ft, domain);
	 cs.setRelativeAccuracy(desiredAccuracy);
	 cs.setEps(eps);
	 cs.setVerbosityLevel(2);
	 cs.setMaxSteps(maxSteps);
	 cs.setIterativeRefinements(iterativeRefinements);
	 cs.setDesiredContraction(desContr);
	 cs.setRelaxedDesiredContraction(relDesContr);
	 cs.setMaximalContraction(maxContr);*/

	auto path = Spacy::PathFollowing::ClassicalContinuation(
			initialNormalComplianceParameter, maxNormalComplianceParameter);

	std::function<
			std::tuple<bool, ::Spacy::Vector, ::Spacy::Real>(
					const ::Spacy::Vector & x, ::Spacy::Real lambda,
					const ::Spacy::Real & theta)> solveFunction =
			[desiredAccuracy, eps, maxSteps, iterativeRefinements, desContr, relDesContr, maxContr, &fn, &ft, &domain] (const ::Spacy::Vector & x, ::Spacy::Real lambda, const ::Spacy::Real & theta) mutable
			{

				using boost::typeindex::type_id_with_cvr;

				// Necessary because ft is casted to Spacy::C2Functional
				ft.updateParam(get(lambda));
				fn.updateParam(get(lambda));

				auto cs = Spacy::CompositeStep::AffineCovariantSolver(fn, ft, domain);

				cs.setRelativeAccuracy(desiredAccuracy);
				cs.setEps(eps);
				cs.setVerbosityLevel(2);
				cs.setMaxSteps(maxSteps);
				cs.setIterativeRefinements(iterativeRefinements);
				cs.setDesiredContraction(desContr);
				cs.setRelaxedDesiredContraction(relDesContr);
				cs.setMaximalContraction(maxContr);
				/*auto bol = cs.getNormalFunctional().returnFunctions();*/
//				std::cout << "Type: " << type_id_with_cvr<decltype(bol)>().pretty_name() << std::endl;
				/*		cs.getTangentialFunctional().paramUpdate(Spacy::get(lambda));*/
				return cs.solvePath(x, lambda, theta );
			};

	std::function<void(const ::Spacy::Vector & solution, unsigned int step) > plotFunction =  [&gridManager,&x, order ] (const ::Spacy::Vector & solution, unsigned int step)
		{
		    Spacy::Kaskade::copy(solution, x);
		    IoOptions options;
			options.outputType = IoOptions::ascii;
			std::string name = "Deformation_" + std::to_string(step);
			std::string outfilename(name);
			writeVTKFile(gridManager.grid().leafGridView(), x, outfilename, options,
					order);
		};

	path.setSolver(solveFunction);
	path.setPlot(plotFunction);

	auto result = path.solve(Spacy::zero(domain));

//		std::cout << "start solver" << std::endl;
//		using namespace std::chrono;
//		auto startTime = high_resolution_clock::now();
//		auto result = cs();
//		std::cout << "computation time: " << duration_cast < seconds
//				> (high_resolution_clock::now() - startTime).count() << "s."
//						<< std::endl;
//
	Spacy::Kaskade::copy(result, x);
//
//	IoOptions options;
//	options.outputType = IoOptions::ascii;
//	std::string name = "Deformation";
//	std::string outfilename(name);
//	writeVTKFile(gridManager.grid().leafGridView(), x, outfilename, options,
//			order);

}


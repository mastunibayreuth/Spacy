#include <iostream>

#include <dune/grid/config.h>
#include <dune/grid/uggrid.hh>

// #include <dune/grid/io/file/gmshreader.hh>

#define SPACY_ENABLE_LOGGING
#include <Spacy/zeroVectorCreator.hh>
#include <Spacy/Adapter/kaskade.hh>
#include <Spacy/Algorithm/Newton/newton.hh>
#include <Spacy/Util/cast.hh>
#include <Spacy/inducedScalarProduct.hh>

#include "mg/hb.hh"
#include "mg/multigrid.hh"
#include "mg/additiveMultigrid.hh"
//#include "linalg/icc0precond.hh"
#include "linalg/direct.hh"
#include <fem/gridmanager.hh>
#include <fem/lagrangespace.hh>
#include <fem/variables.hh>
#include <io/vtk.hh>
#include <utilities/gridGeneration.hh> 
#include <Spacy/Adapter/Kaskade/preconditioner.hh>
// #include "../fung_functional.hh"

#include <Spacy/Algorithm/ACR/acr.hh>

#include <fung/fung.hh>
//#include <fung/examples/biomechanics/adipose_tissue_sommer_holzapfel.hh>
#include <fung/examples/rubber/neo_hooke.hh>
#include <fung/examples/rubber/mooney_rivlin.hh>

#include <fem/particle.hh>
#include "nonlinear_elasticity.hh"
#include "parameters.hh"
//#include <Spacy/Algorithm/Newton/newton.hh>
#include <Spacy/Util/Mixins/Get.hh>

#include "setup.hh"

using namespace Kaskade;

int main()
{

	constexpr int dim = 3;
	const std::string parFile("nonlinear_elasticity.parset");

	ProblemParameters parameters = readParameters(parFile);

	const int refinements = parameters.refinements;
	const int order = parameters.order;
	const bool onlyLowerTriangle = parameters.onlyLowerTriangle;
	const double boundary_force_z = parameters.boundary_force_z;

	const double lambda = parameters.lambda;
	const double mu = parameters.mu;
	const double d = parameters.d;
	const double a = -d;
	const double b = 0.5 * (lambda - d * (4.0));
	const double c = mu + d;

	const double tauInit = parameters.tauInit;
	const double tauMin = parameters.tauMin;
	const double tauIncrease = parameters.tauIncrease;
	const double tauDamp = parameters.tauDamp;

	const int maxSteps = parameters.maxSteps;

	const bool nonlinUpdate = parameters.nonlinUpdate;
	const bool fantasyMaterial = parameters.fantasyMaterial;
	const bool lubkollMaterial = parameters.lubkollMaterial;

	using Grid = Dune::UGGrid<dim>;
	using Scalar = double;
	using H1Space = FEFunctionSpace<ContinuousLagrangeMapper<double,typename Grid::LeafGridView> >;
	using Spaces = boost::fusion::vector<H1Space const*>;
	using VariableDescriptions = boost::fusion::vector< Variable<SpaceIndex<0>,Components<dim>,VariableId<0> > >;
	using VariableSetDescription = VariableSetDescription<Spaces,VariableDescriptions>;
	using VariableSet = typename VariableSetDescription::VariableSet;
	using Vector = Dune::FieldVector<Scalar, dim>;
	using CoefficientVectors = VariableSetDescription::CoefficientVectorRepresentation<0,1>::type;
	const Vector boundary_force
	{ 0., 0., boundary_force_z };

        using PlotSpace = FEFunctionSpace<ContinuousLagrangeMapper<double,typename Grid::LeafGridView> >;
        using PlotSpaces = boost::fusion::vector<H1Space const*,PlotSpace const*>;
        using PlotVariableDescriptions = boost::fusion::vector< Variable<SpaceIndex<0>,Components<dim>,VariableId<0> >,Variable<SpaceIndex<1>,Components<dim>,VariableId<1> > >;
        using PlotVariableSetDescription =::Kaskade::VariableSetDescription<PlotSpaces,PlotVariableDescriptions>;
        using PlotVariableSet = typename PlotVariableSetDescription::VariableSet;

	Dune::FieldVector<double, dim> c0(0.0), dc(1.0);

	//GridManager<Grid> gridManager(createCuboid<Grid>(c0, dc, 1.0, true));
	auto grid = createDefaultCuboidFactory2<Grid>();

	GridManager<Grid> gridManager(std::move(grid));
	gridManager.globalRefine(refinements);

	// Spaces
	H1Space deformationSpace(gridManager, gridManager.grid().leafGridView(),
			order);
	Spaces spaces(&deformationSpace);
	VariableSetDescription description(spaces,
        { "y" });


        PlotSpace plotSpace(gridManager, gridManager.grid().leafGridView(),
                        order);
        PlotSpaces plotSpaces(&deformationSpace, &plotSpace);
        PlotVariableSetDescription plotSetDescription(plotSpaces,
        { "y","o" });
        PlotVariableSet plotSet(plotSetDescription);




	using Matrix = Dune::FieldMatrix<double,dim,dim>;
	auto y0 = FunG::LinearAlgebra::unitMatrix<Matrix>();



 //     y0 = 0.0;
//      std::cout << "Parameter: " << a << " " << b << " " << c << " " << std::endl;

//      auto integrand = FunG::myMaterial<Matrix>(a, b, c, d, y0);
//	auto integrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
//			Matrix>(0.08625, 0.08625, 0.68875, -1.895, y0);

//   auto integrand = FunG::myCompressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,Matrix>(a, b, c, d, y0);

//	auto integrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
//			Matrix>(100, 100, 1, -1.0, y0);


	 auto integrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>,
	FunG::LN,Matrix>(3.690000000,  0.410000000,   2.090000000,  -13.20000000,
	y0);


//	if (lubkollMaterial)
//	{
//		integrand = FunG::compressibleMooneyRivlin<FunG::Pow<2>, FunG::LN,
//				Matrix>(0.08625, 0.08625, 0.68875, -1.895, y0);
//	}
//      auto integrand = FunG::incompressibleNeoHooke(1.0,y0);

	using boost::typeindex::type_id_with_cvr;
//      std::cout << "IntegrandType:" <<  type_id_with_cvr<decltype(integrand)>().pretty_name() << std::endl;

	using Functional = NonlinearElasticityFunctional<decltype(integrand),VariableSetDescription>;
	using Assembler = VariationalFunctionalAssembler<LinearizationAt<Functional> >;
	using Operator = AssembledGalerkinOperator<Assembler>;

	auto F = Functional(integrand, boundary_force);
	Assembler assembler(gridManager, spaces);

	VariableSet y(description);
	VariableSet x(description);

	assembler.assemble(linearization(F, y));

	CoefficientVectors rhs(assembler.rhs());
	CoefficientVectors solution(
			VariableSetDescription::CoefficientVectorRepresentation<0, 1>::init(
					spaces));

	ParticleCloud<dim, Grid,
			decltype(gridManager.grid().leafGridView().indexSet())> cloud(
			gridManager.grid());

        cloud.checkNeighborSearch(900);

	 auto u = x;
	 u = 0;
	 const auto & particles = cloud.getParticles();
	 auto count = 0;

        std::for_each(std::begin(boost::fusion::at_c<0>(u.data).coefficients()),
                        std::end(boost::fusion::at_c<0>(u.data).coefficients()),
                        [particles,count] (auto & entry) mutable
                        {

                                entry = particles[count].isNeighbor_;

                                count++;

                        });

        IoOptions options2;
        options2.outputType = IoOptions::ascii;

        writeVTKFile(gridManager.grid().leafGridView(), u, "isNeighbor", options2,
                        1);


//


//	y = 0.5;
//	cloud.adaptiveMoveAlongVectorField(boost::fusion::at_c<0>(y.data),tauInit, tauMin, tauIncrease, tauDamp, maxSteps);
//	IoOptions options;
//	    options.outputType = IoOptions::ascii;
//
//
//
//	auto count = 0;
//	auto u = y;
//	u = 0;
//	const auto & particles = cloud.getParticles();
//
//							std::for_each(std::begin(boost::fusion::at_c<0>(u.data).coefficients()),
//									std::end(boost::fusion::at_c<0>(u.data).coefficients()),
//									[particles,count] (auto & entry) mutable
//									{
//
//										if(!(particles[count].isNonlinear_))
//										{
//											entry = 1.0;
//										}
//
//										count++;
//
//									});
//
//							writeVTKFile(gridManager.grid().leafGridView(), u, "lin", options, 1);
//
//	cloud.adaptiveMoveAlongVectorField(boost::fusion::at_c<0>(y.data),tauInit, tauMin, tauIncrease, tauDamp, maxSteps);
//
//	cloud.updateGlobalDisplacements(y);
//
//
//		std::string name = "TestMoveAlong";
//	//
//		writeVTKFile(gridManager.grid().leafGridView(), y, name, options, 1);








	auto domain = Spacy::Kaskade::makeHilbertSpace<VariableSetDescription>(
			description);
	auto range = Spacy::Kaskade::makeHilbertSpace<VariableSetDescription>(
			description);

	using FType = Spacy::Kaskade::C2Functional<Functional>;
	using FLin = typename Spacy::Kaskade::C2Functional<Functional>::Linearization;

	using X = Dune::BlockVector<Dune::FieldVector<double,dim>>;
	using NMatrix = NumaBCRSMatrix<Dune::FieldMatrix<double,dim,dim>>;

	NMatrix Amat(assembler.template get<0, 0>(), true);

	auto mg = makeBPX(Amat, gridManager);

	std::function<Spacy::LinearSolver(const FLin&)> precond =
			[&mg,&gridManager](const FLin& f)
			{

				//const auto & dummy = f.get();
				//auto bcrs = dummy.template toBCRS<3>();
//				auto NumAMatrix_ = NumaBCRSMatrix<Dune::FieldMatrix<double, dim, dim>>(*bcrs,true);
				// More Efficient  Create Matrix Outside once and update it
				// avoid Copy somehow  check wether NumAMatrix_ lives   ho
			//	mg = makeBPX( NumaBCRSMatrix<Dune::FieldMatrix<double, dim, dim>>(*bcrs,true), gridManager);
				return Spacy::makeTRCGSolver(f,Spacy::Kaskade::makePreconditioner<typename FType::VariableSetDescription,typename FType::VariableSetDescription>(f.range(),f.domain(),mg));
			};

	CoefficientVectors coeff(
			VariableSetDescription::template CoefficientVectorRepresentation<>::init(
					spaces));

	// copy Y
	auto nonLinUpdateFunc =
                        [&cloud,&y, &x,&coeff,&gridManager,&plotSet,tauInit, tauMin, tauIncrease, tauDamp, maxSteps] (const ::Spacy::Vector& x_, Spacy::Vector& dx_)
			{
                                static int counter = 1;

				IoOptions options;
				options.outputType = IoOptions::ascii;

				Spacy::Kaskade::copy( x_, x);
				Spacy::Kaskade::copy( dx_, y);

				auto lin = y;
                                std::string displacement = "DisplacementLin_" + std::to_string(counter);
                                writeVTKFile(gridManager.grid().leafGridView(), lin, displacement, options,1);

                                lin += x;

				std::string nameLin = "DeformationLin_" + std::to_string(counter);

				// should be x + dx
				writeVTKFile(gridManager.grid().leafGridView(), lin, nameLin, options,
						1);

				cloud.adaptiveMoveAlongVectorField(boost::fusion::at_c<0>(y.data),x,tauInit, tauMin, tauIncrease, tauDamp, maxSteps);
                                //cloud.moveAlongVectorField(boost::fusion::at_c<0>(y.data),x, 1, 1);
				//cloud.outputParticles(nameParticles);
				// update not displacement
				cloud.updateGlobalDisplacements(y);

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


//				std::string name = "OrientationChange" + std::to_string(counter);

//				writeVTKFile(gridManager.grid().leafGridView(), par, name, options,
//						1);

//				name = "Deformation_" + std::to_string(counter);

//				writeVTKFile(gridManager.grid().leafGridView(), y, name, options,
//						1);



				//component<0>(u) = component<0>(solution);
				boost::fusion::at_c<0>(coeff.data) = boost::fusion::at_c<0>(y.data).coefficients();
				Spacy::Kaskade::copyFromCoefficientVector<VariableSetDescription> (coeff,dx_);

                                cloud.updateDisplacements(y);

                                y += x;
                                component<0>(plotSet) = component<0>(y);
                                component<1>(plotSet) = component<0>(par);

                                std::string name = "Deformation_" + std::to_string(counter);
                                writeVTKFile(gridManager.grid().leafGridView(), plotSet, name, options,
                                                                                1);

				counter++;
			};

	auto output =
			[&y,&gridManager] (const Spacy::Vector& x_)
			{

				static int counter = 0;

				auto x = y;
				IoOptions options;
				options.outputType = IoOptions::ascii;

				Spacy::Kaskade::copy( x_, x);

				std::string nameLin = "DeformationActualUpdate: " + std::to_string(counter);

				writeVTKFile(gridManager.grid().leafGridView(), x, nameLin, options,
						1);
				counter++;

			};

	auto f = Spacy::Kaskade::makeC2Functional(F, domain);

	// kaskade test variable for nonlinupdate
	f.setSolverCreator(precond);
	Spacy::ACR::ACRSolver solver(f);
	if (nonlinUpdate)
	{
		solver.setNonlinUpdate(nonLinUpdateFunc);
	}
	solver.setOutput(output);
	auto result = solver();
	Spacy::Kaskade::copy(result, y);
	IoOptions options;
	options.outputType = IoOptions::ascii;
	std::string name = "DeformationFinal";

	writeVTKFile(gridManager.grid().leafGridView(), y, name, options, 1);





	return 0;
}

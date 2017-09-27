#pragma once

#include <dune/grid/config.h>
#include <dune/grid/uggrid.hh>

#include "fem/gridmanager.hh"
#include "mg/hb.hh"
#include "mg/multigrid.hh"
#include "mg/additiveMultigrid.hh"
#include <fem/variables.hh>
#include <io/vtk.hh>

#include <fung/fung.hh>
#include <fung/examples/rubber/neo_hooke.hh>

#include <Spacy/Adapter/Kaskade/preconditioner.hh>
#include <Spacy/Algorithm/ACR/acr.hh>

#include "cuboid.hh"
#include "functional.hh"
#include "parameters.hh"
// enum class Dimension { DIM = 3};
// Dune::UGGrid<int(Dimension::DIM)>
// dim als template



template<class Grid>
std::unique_ptr<Grid> createGridFactory(int numberOfCubes, int dim)

{
    // By Tobias Sproll
    std::vector<cuboid> cubes(0);
    for (int i = 0; i < numberOfCubes; i++)
    {
                cuboid myCuboid(0.0, 0.0+i, 0.0, 1., 1., 1.);
                cubes.push_back(myCuboid);
    }
    
    cuboid myCuboid(cubes);
    std::vector<Dune::FieldVector<double,3 > > vertices =myCuboid.getVertices();
    std::vector<std::vector<unsigned int> > tetraeder =myCuboid.getTetraeder();

    Dune::GridFactory<Grid> factory;
    Dune::GeometryType gt(Dune::GeometryType::simplex,dim);

    for(int i = 0; i < vertices.size(); i++)
    {
        factory.insertVertex(vertices[i]);
    }
    
    for(int i = 0; i < tetraeder.size(); i++)
    {
        factory.insertElement(gt,tetraeder[i]);
    }
    
    return std::unique_ptr<Grid>(factory.createGrid());   

}
// replace auto
template <int dim,  class Parameters, class Grid, class GridManager, class Vector>
auto setupFactory( const Parameters & parameters, GridManager & gridManager, const Vector & boundary_force ) 
{
    using namespace Kaskade;
    
    const int order       = parameters.order;
    const bool onlyLowerTriangle = parameters.onlyLowerTriangle;
    
    using Scalar = double;
    using H1Space = FEFunctionSpace<ContinuousLagrangeMapper<double,typename Grid::LeafGridView> >;
    using Spaces = boost::fusion::vector<H1Space const*>;
    using VariableDescriptions = boost::fusion::vector< Variable<SpaceIndex<0>,Components<dim>,VariableId<0> > >;
    using VariableSetDescription = VariableSetDescription<Spaces,VariableDescriptions>;
    using VariableSet = typename VariableSetDescription::VariableSet;
    
    H1Space deformationSpace(gridManager,gridManager.grid().leafGridView(),order);

    Spaces spaces(&deformationSpace);
    VariableSetDescription description(spaces, { "y" });

    using Matrix = Dune::FieldMatrix<double,dim,dim>;
    
    auto y0 = FunG::LinearAlgebra::unitMatrix<Matrix>();
    auto integrand = FunG::incompressibleNeoHooke(1.0,y0);
    
    using Functional = NonlinearElasticityFunctional<decltype(integrand),VariableSetDescription>;
    using Assembler = VariationalFunctionalAssembler<LinearizationAt<Functional> >;
    
    auto F = Functional(integrand,  boundary_force);
    Assembler assembler(gridManager,spaces);
    
    VariableSet y(description);
     
     // Check if necessary
    assembler.assemble(linearization(F,y));

     //  Spacy
     
    auto domain = Spacy::Kaskade::makeHilbertSpace<VariableSetDescription>(spaces, {0u});
    auto fn = Spacy::Kaskade::makeC2Functional( F, domain );
    
    using FType = Spacy::Kaskade::C2Functional<Functional>;
    using FLin = typename Spacy::Kaskade::C2Functional<Functional>::Linearization;

    using X = Dune::BlockVector<Dune::FieldVector<double,dim>>;
    using NMatrix = NumaBCRSMatrix<Dune::FieldMatrix<double,dim,dim>>;
    
    NMatrix Amat(assembler.template get<0,0>(),true);

    auto mg = makeBPX(Amat,gridManager);

    std::function<Spacy::LinearSolver(const FLin&)>
    precond = [&mg](const FLin& f)
    {
        return Spacy::makeTCGSolver(f,Spacy::Kaskade::makePreconditioner<typename FType::VariableSetDescription,typename FType::VariableSetDescription>(f.range(),f.domain(),mg));
    };
    
    // compute solution by ACR
      
    fn.setSolverCreator(precond);
    Spacy::ACR::ACRSolver solver(fn);
    auto result = solver();

    VariableSet x(description);
    Spacy::Kaskade::copy(result,x);

    IoOptions options;
    options.outputType = IoOptions::ascii;
    std::string outfilename("ReferenceDeformation");
    writeVTKFile(gridManager.grid().leafGridView(),x,outfilename,options,1);
    
    return x;
}
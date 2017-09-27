#pragma once
#include <string>
#include <dune/common/parametertree.hh>                                                                                                                                          
#include <dune/common/parametertreeparser.hh>  

struct ProblemParameters
{
	unsigned order;
	unsigned refinements;

	bool onlyLowerTriangle;

	double lambda;
	double mu;

	double d;
	double boundary_force_z;

	double tauInit;
	double tauMin;
	double tauIncrease;
	double tauDamp;

	int maxSteps;

	bool nonlinUpdate;
	bool fantasyMaterial;
	bool lubkollMaterial;

};

// ProblemParameters readParameters(const std::string & filename);

ProblemParameters readParameters(const std::string & filename)
{
	ProblemParameters par;

	Dune::ParameterTree parTree;
	Dune::ParameterTreeParser::readINITree(filename, parTree);

	par.order = parTree.get<unsigned>("order");
	par.boundary_force_z = parTree.get<double>("boundary_force_z");
	par.onlyLowerTriangle = parTree.get<bool>("onlyLowerTriangle");
	par.refinements = parTree.get<unsigned>("refinements");
	par.lambda = parTree.get<double>("lambda");
	par.mu = parTree.get<double>("mu");
	par.d = parTree.get<double>("d");

	par.tauInit = parTree.get<double>("tauInit");
	par.tauMin = parTree.get<double>("tauMin");
	par.tauIncrease = parTree.get<double>("tauIncrease");
	par.tauDamp = parTree.get<double>("tauDamp");

	par.maxSteps = parTree.get<int>("maxSteps");

	par.nonlinUpdate = parTree.get<bool>("nonlinUpdate");
	par.fantasyMaterial = parTree.get<bool>("fantasyMaterial");
	par.lubkollMaterial = parTree.get<bool>("lubkollMaterial");

	// return mit std::move()
	return par;

}

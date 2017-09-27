#include "parameters.hh"
#include <dune/common/parametertree.hh>                                                                                                                                          
#include <dune/common/parametertreeparser.hh>    



// ProblemParameters readParameters(const std::string & filename)
// {
//     ProblemParameters par;
//     
//     Dune::ParameterTree parTree;
//     Dune::ParameterTreeParser::readINITree(filename,  parTree);
//     
//     par.order = parTree.get<unsigned>("order");
//     par.boundary_force_z = parTree.get<double>("boundary_force_z");
//     par.onlyLowerTriangle = parTree.get<bool>("onlyLowerTriangle");
//     par.refinements = parTree.get<unsigned>("refinements");
//    
//     // return mit std::move()
//     return par;
// 
//     
//     
// } 


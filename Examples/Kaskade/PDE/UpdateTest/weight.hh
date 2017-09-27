#include <utility> // std::move

#include <boost/timer/timer.hpp>

#include "dune/grid/config.h"
#include "dune/grid/uggrid.hh"

#include "fem/assemble.hh"
#include "fem/istlinterface.hh"
#include "fem/lagrangespace.hh"

#include "mg/pcg.hh"
// #include "linalg/iccprecond.hh"

using namespace Kaskade;

// #include "emg.hh"
#include "cuboid.hh"


template <class VarSet, class Space ,class Spaces, class CoefVec>
class Weight{

	public:
		using Grid				= Dune::UGGrid<3>;
		using LeafView			= Grid::LeafGridView;
		using Functional 			= EMGFunctional<double,VarSet>;
		using Assembler	 		= VariationalFunctionalAssembler<LinearizationAt<Functional> >;


		Weight() {}

		static std::unique_ptr<Grid> createGrid(){
			const int dim =3;
			std::vector<cuboid> cubes;
			for (int i = 0; i<10;i++){
				for (int j = 0; j<10;j++){
					for (int k = 0; k<1;k++){
						cuboid myCuboid(-0.05+i*0.01,-0.05+j*0.01,k*0.01,0.01,0.01,0.01);
						cubes.push_back(myCuboid);
					}
				}
			}
			cuboid myCuboid(cubes);
			std::vector<Dune::FieldVector<double,3 > > vertices =myCuboid.getVertices();
			std::vector<std::vector<unsigned int> > tetraeder =myCuboid.getTetraeder();

			Dune::GridFactory<Grid> factory;
			Dune::GeometryType gt(Dune::GeometryType::simplex,dim);

			for(int i=0;i<vertices.size();i++){
				factory.insertVertex(vertices[i]);
			}
			for(int i=0;i<tetraeder.size();i++){
				factory.insertElement(gt,tetraeder[i]);
			}

			std::unique_ptr<Grid> grid (factory.createGrid());
			return grid;
		}

		static void refineGrid(GridManager<Grid> &gridManager, int refinements, int adaptive){
			gridManager.globalRefine(refinements);
			typedef typename LeafView::template Codim <0>::Iterator ElementLeafIterator;

			for (int i =1; i<=adaptive; i++){
				LeafView leafView = gridManager.grid().leafGridView();
				for (ElementLeafIterator it = leafView.template begin <0>(); it!=leafView.template end <0>(); ++it){
					if ((it->geometry().corner(0)[2] >  0.009 || it->geometry().corner(1)[2] >  0.009 || it->geometry().corner(2)[2] >  0.009 || it->geometry().corner(3)[2] >  0.009) &&
						(it->geometry().corner(0)[0] < 	0.041 || it->geometry().corner(1)[0] <  0.041 || it->geometry().corner(2)[0] <  0.041 || it->geometry().corner(3)[0] <  0.041) &&
						(it->geometry().corner(0)[0] > -0.04 || it->geometry().corner(1)[0] > -0.04 || it->geometry().corner(2)[0] > -0.04 || it->geometry().corner(3)[0] > -0.04) &&
						(it->geometry().corner(0)[1] <  0.008 || it->geometry().corner(1)[1] <  0.008 || it->geometry().corner(2)[1] <  0.008 || it->geometry().corner(3)[1] <  0.008) &&
						(it->geometry().corner(0)[1] > -0.004 || it->geometry().corner(1)[1] > -0.004 || it->geometry().corner(2)[1] > -0.004 || it->geometry().corner(3)[1] > -0.004)){
						gridManager.mark(1,*it);
					}
				}
				gridManager.adaptAtOnce();

				gridManager.flushMarks();
			}

			std::cout << std::endl << "Grid was created successfully" << std::endl;
			std::cout << "Grid: " << gridManager.grid().size(0) << " tetrahedra, " << std::endl;
			std::cout << "      " << gridManager.grid().size(1) << " edges, " << std::endl;
			std::cout << "      " << gridManager.grid().size(2) << " points" << std::endl;


		}


		static void computeWeight(typename VarSet::VariableSet& w,  GridManager<Grid> &gridManager ,int order, double x, double y){

			int 	onlyLowerTriangle	=	true,
					verbosity		=	2,
					iteSteps 		=	1000;
			double	iteEps 		=	1e-10,
					dropTol 		= 	0.01;

			Space potentialSpace(gridManager,gridManager.grid().leafGridView(),order);
			Spaces spaces(&potentialSpace);
			std::string varNames[1] = { "u" };
			VarSet variableSet(spaces,varNames);
			typename VarSet::VariableSet u(variableSet);


			int const nvars	= EMGFunctional<double,VarSet>::AnsatzVars::noOfVariables;
			int const neq 	= EMGFunctional<double,VarSet>::TestVars::noOfVariables;

			std::cout << std::endl << "no of variables = " << nvars << std::endl;
			std::cout << "no of equations = " << neq   << std::endl;
			size_t dofs = variableSet.degreesOfFreedom(0,nvars);
			std::cout << "number of degrees of freedom = " << dofs << std::endl;

			CoefVec solution(VarSet::template CoefficientVectorRepresentation<0,1>::init(spaces));
			Assembler assembler(gridManager,spaces);

			solution=0;
			Functional F(x,y);

			assembler.assemble(linearization(F,u));
			CoefVec rhs(assembler.rhs());

			AssembledGalerkinOperator<Assembler,0,neq,0,nvars> A(assembler, onlyLowerTriangle);

			boost::timer::cpu_timer iteTimer;

			Dune::InverseOperatorResult res;

			ICCPreconditioner<AssembledGalerkinOperator<Assembler,0,neq,0,nvars> > icc(A,dropTol);
			NMIIIPCGSolver<CoefVec> pcg(A,icc,iteEps,iteSteps,verbosity);
			pcg.apply(solution,rhs,res);
			solution *= -1.0;
			w.data = solution.data;


			std::cout << "iterative solve eps= " << iteEps << ": "
					  << (res.converged?"converged":"failed") << " after "
					  << res.iterations << " steps, rate="
					  << res.conv_rate << ", computing time=" << (double)(iteTimer.elapsed().user)/1e9 << "s\n";


		}



	private:


};

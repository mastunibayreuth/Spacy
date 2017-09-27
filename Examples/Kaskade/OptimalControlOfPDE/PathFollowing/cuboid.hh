#include <vector>


class cuboid{
public:
	std::vector<Dune::FieldVector<double,3> > vertices;
	std::vector<std::vector<unsigned int> > tetraeder;

	
	//creates a unit cuboid with centre at p=(0,0,0)
	cuboid(){
		Dune::FieldVector<double,3> v; 
		std::vector<unsigned int> p(4,0);

		//points in the frontside
		v[0]=-0.5;	v[1]=-0.5; v[2]=-0.5;	this->vertices.push_back(v);
		v[0]=0.5;	v[1]=-0.5; v[2]=-0.5;	this->vertices.push_back(v);
		v[0]=0.5;	v[1]=-0.5; v[2]=0.5;	this->vertices.push_back(v);
		v[0]=-0.5;	v[1]=-0.5; v[2]=0.5;	this->vertices.push_back(v);

		//points in the backside
		v[0]=-0.5;	v[1]=0.5; 	v[2]=-0.5;	this->vertices.push_back(v);
		v[0]=0.5; 	v[1]=0.5; 	v[2]=-0.5;	this->vertices.push_back(v);
		v[0]=0.5;	v[1]=0.5; 	v[2]=0.5; 	this->vertices.push_back(v);
		v[0]=-0.5;	v[1]=0.5;	v[2]=0.5; 	this->vertices.push_back(v);

		//point in the centre
		v[0]=0; 	v[1]=0;		v[2]=0;		this->vertices.push_back(v);

		//frontside
		p[0]=0; p[1]=1; p[2]=2; p[3]=8;		this->tetraeder.push_back(p);
		p[0]=0; p[1]=2; p[2]=3; p[3]=8; 	this->tetraeder.push_back(p);

		//backside
		p[0]=4; p[1]=5; p[2]=6; p[3]=8; 	this->tetraeder.push_back(p);
		p[0]=4; p[1]=6; p[2]=7; p[3]=8; 	this->tetraeder.push_back(p);

		//rightside
		p[0]=1; p[1]=5; p[2]=6; p[3]=8; 	this->tetraeder.push_back(p);
		p[0]=1; p[1]=6; p[2]=2; p[3]=8; 	this->tetraeder.push_back(p);

		//leftside
		p[0]=0; p[1]=4; p[2]=7; p[3]=8; 	this->tetraeder.push_back(p);
		p[0]=0; p[1]=7; p[2]=3; p[3]=8; 	this->tetraeder.push_back(p);

		//bottom
		p[0]=0; p[1]=1; p[2]=5; p[3]=8; 	this->tetraeder.push_back(p);
		p[0]=0; p[1]=5; p[2]=4; p[3]=8; 	this->tetraeder.push_back(p);

		//top
		p[0]=3; p[1]=2; p[2]=6; p[3]=8; 	this->tetraeder.push_back(p);
		p[0]=3; p[1]=6; p[2]=7; p[3]=8; 	this->tetraeder.push_back(p);
	}

	
	//creates an unit cuboid with the left,down front corner in the point(x,y,z) and length (l_x,l_y,l_z)  
	cuboid(double x ,double y, double z,double l_x, double l_y, double l_z){
		Dune::FieldVector<double,3> v; 
		std::vector<unsigned int> p(4,0);

		//points in the frontside
		v[0]=x;		v[1]=y; v[2]=z;			this->vertices.push_back(v);
		v[0]=x+l_x;	v[1]=y; v[2]=z;			this->vertices.push_back(v);
		v[0]=x+l_x;	v[1]=y; v[2]=z+l_z;		this->vertices.push_back(v);
		v[0]=x;		v[1]=y; v[2]=z+l_z; 	this->vertices.push_back(v);

		//points in the backside
		v[0]=x; 	v[1]=y+l_y; v[2]=z;		this->vertices.push_back(v);
		v[0]=x+l_x; v[1]=y+l_y; v[2]=z;		this->vertices.push_back(v);
		v[0]=x+l_x; v[1]=y+l_y; v[2]=z+l_z; this->vertices.push_back(v);
		v[0]=x;		v[1]=y+l_y; v[2]=z+l_z; this->vertices.push_back(v);

		//point in the center
		v[0]=x+0.5*l_x; v[1]=y+0.5*l_y ;v[2]=z+0.5*l_z; this->vertices.push_back(v);

		//frontside
		p[0]=0; p[1]=1; p[2]=2; p[3]=8; this->tetraeder.push_back(p);
		p[0]=0; p[1]=2; p[2]=3; p[3]=8; this->tetraeder.push_back(p);

		//backside
		p[0]=4; p[1]=5; p[2]=6; p[3]=8; this->tetraeder.push_back(p);
		p[0]=4; p[1]=6; p[2]=7; p[3]=8; this->tetraeder.push_back(p);

		//rightside
		p[0]=1; p[1]=5; p[2]=6; p[3]=8; this->tetraeder.push_back(p);
		p[0]=1; p[1]=6; p[2]=2; p[3]=8; this->tetraeder.push_back(p);

		//leftside
		p[0]=0; p[1]=4; p[2]=7; p[3]=8; this->tetraeder.push_back(p);
		p[0]=0; p[1]=7; p[2]=3; p[3]=8; this->tetraeder.push_back(p);

		//bottom
		p[0]=0; p[1]=1; p[2]=5; p[3]=8; this->tetraeder.push_back(p);
		p[0]=0; p[1]=5; p[2]=4; p[3]=8; this->tetraeder.push_back(p);

		//top
		p[0]=3; p[1]=2; p[2]=6; p[3]=8; this->tetraeder.push_back(p);
		p[0]=3; p[1]=6; p[2]=7; p[3]=8; this->tetraeder.push_back(p);
	}
			
	//return a list with the vertices of the cuboid
	std::vector<Dune::FieldVector<double,3> >& getVertices(){return this->vertices;}
	//return a list with the tetraeders of the cuboid
	std::vector<std::vector<unsigned int> >& getTetraeder(){return this->tetraeder;}   
	
	//creates a new cuboid with an given vector of qubes
	//vertices that are twice or more will be deleted and the tedrader will be updated
	cuboid(std::vector<cuboid> cubes){
		std::vector<Dune::FieldVector<double,3> > verts;
		std::vector<std::vector<unsigned int> > tet;
		std::vector<unsigned int>  new_ind;
		Dune::FieldVector<double,3> totest;
		Dune::FieldVector<double,3> sub;
		std::vector<int> toerase;
				
		
		for(int k=0;k<cubes.size();k++){
			for(int i=0;i<9;i++){
				verts.push_back(cubes[k].getVertices()[i]);
				new_ind.push_back(k*9+i);
			}
			for(int i=0;i<12;i++){
				for(int j=0;j<4;j++){
					cubes[k].getTetraeder()[i][j]+=k*9;
				}
				tet.push_back(cubes[k].getTetraeder()[i]);
			}
		}
		
		std::vector<bool> erased(verts.size(),false);
				
		
		for(unsigned int i=0;i<verts.size();i++){
			totest=verts[i];
			for(unsigned int j=i+1;j<verts.size();j++){
				sub=totest-verts[j];
				if(sub.two_norm()<pow(10,-12) && !erased[j]){
					erased[j]=true;
					new_ind[j]=new_ind[i];
					for(int l=j+1;l<new_ind.size();l++){
						if(new_ind[l]>new_ind[j]) new_ind[l]-=1;
					}
					toerase.push_back(j);
				}
			}
		}
		
		for(int k=0;k<tet.size();k++){
			for(int l=0;l<4;l++){
				tet[k][l]=new_ind[tet[k][l]];
			}
		}
		
		sort(toerase.begin(),toerase.end());
		for(int k=toerase.size()-1;k>=0;k--){
			verts.erase(verts.begin()+toerase[k]);
		}
		
		this->vertices=verts;
		this->tetraeder=tet;  
	}
	
	
};

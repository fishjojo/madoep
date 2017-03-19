/*
  This file is part of MADNESS.

  Copyright (C) 2007,2010 Oak Ridge National Laboratory

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  For more information please contact:

  Robert J. Harrison
  Oak Ridge National Laboratory
  One Bethel Valley Road
  P.O. Box 2008, MS-6367

  email: harrisonrj@ornl.gov
  tel:   865-241-3937
  fax:   865-572-0680
*/

//#define WORLD_INSTANTIATE_STATIC_TEMPLATES

/*!
  \file examples/oep.cc
  \brief optimized effective potentials for DFT
*/

#include <chem/mp2.h>
#include <chem/SCFOperators.h>

#include <madness/mra/operator.h>
#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include <madness/mra/lbdeux.h>
#include <madness/world/worldmem.h>

#if defined(HAVE_SYS_TYPES_H) && defined(HAVE_SYS_STAT_H) && defined(HAVE_UNISTD_H)
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
static inline int file_exists(const char * inpname)
{
    struct stat buffer;
    int rc = stat(inpname, &buffer);
    return (rc==0);
}
#endif
using namespace madness;

//static const double   rcut = 0.01; // Smoothing distance in 1e potential
//static const double d12cut = 0.01; // Smoothing distance in wave function
static long ngrid=400;

typedef Tensor<double> tensorT;

static double rr(const coord_3d& r) {
	return sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
}

// Smoothed 1/r potential (c is the smoothing distance)
static double u(double r, double c) {
    r = r/c;
    double r2 = r*r, pot;
    if (r > 6.5){
        pot = 1.0/r;
    } else if (r > 1e-2) {
        pot = erf(r)/r + exp(-r2)*0.56418958354775630;
    } else{
        pot = 1.6925687506432689-r2*(0.94031597257959381-r2*(0.39493270848342941-0.12089776790309064*r2));
    }

    return pot/c;
}

template<size_t NDIM>
void draw_line(World& world, Function<double,NDIM>& pair, const std::string restart_name) {

    Vector<double,NDIM> lo(0.0), hi(0.0);
    lo[1]=-8.0;
    hi[1]=8.0;

    {
        std::string filename="line_"+restart_name;
        trajectory<NDIM> line=trajectory<NDIM>::line2(lo,hi,601);
        plot_along<NDIM>(world,line,pair,filename);
    }

}


template<size_t NDIM>
void save_function(World& world, Function<double,NDIM>& pair, const std::string name) {
    pair.print_size("saving function "+name);
    archive::ParallelOutputArchive ar(world, name.c_str());
    ar & pair;// & cell;
}


template<size_t NDIM>
void load_function(World& world, Function<double,NDIM>& pair, const std::string name) {
    if (world.rank()==0)  print("loading function ", name);
    archive::ParallelInputArchive ar(world, name.c_str());
    ar & pair;// & cell;
    pair.print_size("loaded function "+name);
}

struct LBCost {
    double leaf_value;
    double parent_value;
    LBCost(double leaf_value=1.0, double parent_value=1.0)
        : leaf_value(leaf_value)
        , parent_value(parent_value)
    {}

    double operator()(const Key<6>& key, const FunctionNode<double,6>& node) const {
        if (key.level() <= 1) {
            return 100.0*(leaf_value+parent_value);
        }
        else if (node.is_leaf()) {
            return leaf_value;
        }
        else {
            return parent_value;
        }
    }
};


static Tensor<double> read_radial_grid(int k, std::string filename) {

    FILE* file = fopen(filename.c_str(),"r");

    Tensor<double> a(k);
    for (int i=0; i<k; ++i) {
    	double c;
    	if (fscanf(file,"%lf",&c) != 1) {
    		print("read_grid: failed reading data from file", filename);
    		MADNESS_EXCEPTION("",0);
    	}
    	a(i) = c;
    }
    return a;
}


/// xyz coord file starts with two lines of meta-data: # atoms and a comment
/*static std::vector<std::pair<coord_3d,double> > read_xyz_coord(std::string filename) {

	typedef std::pair<coord_3d, double> Atom;

    std::ifstream file(filename.c_str());
    std::string line;

    long k;
    if (not (std::getline(file,line))) MADNESS_EXCEPTION("failed reading 1st line of coord data",0);
    if (not (std::istringstream(line) >> k)) MADNESS_EXCEPTION("failed reading k",0);
    if (not (std::getline(file,line))) MADNESS_EXCEPTION("failed reading 2nd line of coord data",0);

    print("reading ",k,"coordinates from file",filename);

    std::vector<Atom> atoms;
    int i=0;
	double x,y,z;
    std::string element;

    while (file >> element >> x >> y >> z) {
    	MADNESS_ASSERT(element=="N");
    	Atom atom;
    	if (element=="N") atom.second=7.0;
    	atom.first[0]=x;
    	atom.first[1]=y;
    	atom.first[2]=z;
    	atoms.push_back(atom);
    	print("read",atom.second,atom.first);

    	// exit if we have read enough data
    	if ((++i) ==k) break;
    }
    MADNESS_ASSERT(i==k);
    file.close();
    return atoms;
}*/


struct molecular_potential {
	typedef std::pair<coord_3d, double> Atom;

	std::vector<Atom> atoms;
	molecular_potential() {};
	molecular_potential(std::vector<Atom>& atoms) : atoms(atoms) {}
	double operator()(const coord_3d& xyz) const {
		double val=0.0;
		std::vector<Atom>::const_iterator it;
		for (it=atoms.begin(); it!=atoms.end(); it++) {
			const Atom& atom=*it;
			val+=atom.second*u(rr(xyz-atom.first),0.00001);
		}
		return val;
	}
};


/// wrapper class for the optimized effective potential

/// The potential is given on a radial grid. Interpolate if necessary
struct recpot {

	/// the number of grid points
	long k;

	/// the effective potential tabulated
	tensorT potential;

	/// the nuclear charge
	double ZZ;

	/// ctor with the nuclear charge to subtract for better interpolation

	/// @param[in]	file_grid   name of the file with grid points
	/// @param[in]	file_pot1   name of the file with the potential on the grid points
	/// @param[in]	file_pot2   name of the other file with the potential on the grid points
	/// @param[in]	ZZ          nuclear charge (unused at the moment)
	recpot(const std::string file_grid, const std::string file_pot1, const std::string file_pot2,
					const long ZZ) : ZZ(double(ZZ)) {

		// number of grid points
		k=ngrid;

		// read the grid points and the potential values
		potential=tensorT(2,k);
		// this is an assignment of a slice
		potential(0,_)=read_radial_grid(k,file_grid);
		potential(1,_)=read_radial_grid(k,file_pot1);
		potential(1,_)+=read_radial_grid(k,file_pot2);

		// subtract the nuclear potential; must be consistent with operator()
		for (int i=0; i<k; ++i) {
			double r=potential(0,i);
			potential(1,i)+=ZZ*u(r,1.e-6);
		}
	}

	/// ctor with the radial density

	/// @param[in]	file_grid 	name of the file with grid points
	/// @param[in]	file_pot1 	name of the file with the radial density on the grid points
	recpot(const std::string file_grid, const std::string file_pot1) : ZZ(-1.0) {

		// number of grid points
		k=400;

		// read the grid points and the potential values
		potential=tensorT(2,k);
		// this is an assignment of a slice
		potential(0,_)=read_radial_grid(k,file_grid);
		potential(1,_)=read_radial_grid(k,file_pot1);
	}

	/// return the value of the potential; interpolate if necessary

	/// @param[in]	xyz		cartesian coordinates of a point
	/// @return		the value of the potential at point xyz
	double operator()(const coord_3d& xyz) const {
		double r=rr(xyz);
		if (ZZ>0.0) return (interpolate(r));	// if a potential
//		if (ZZ>0.0) return (interpolate(r)- ZZ* u(r,1.e-6));	// if a potential
		else return (interpolate(r)/(r*r));						// if a radial density
	}

	/// interpolate the radial potential from the grid
	double interpolate(const double& r) const {

		int i=0;
		// upon loop exit i will be the index with the grid point right behind r
		if (r<1.0) i=195;
		if (r<0.3) i=145;
		if (r<0.1) i=100;
		if (r<0.02) i=45;
		for (i=0; i<k; ++i) {
			if (potential(0,i)<r) continue;
			break;
		}

		// fit a linear curve: f(x)=a*x + b
		double delta_y=potential(1,i) - potential(1,i-1);
		double delta_x=potential(0,i) - potential(0,i-1);
		double a=delta_y/delta_x;

		double dx=r-potential(0,i-1);
		double val=potential(1,i-1) + dx*a;
		return val;
	}

};


template<size_t NDIM>
void draw_plane(World& world, Function<double,NDIM>& function, const std::string restart_name) {

    std::string filename="plane_"+restart_name;
    const double scale=0.01;

    // assume a cubic cell
    double hi=FunctionDefaults<NDIM>::get_cell_width()[0]*0.5*scale;
    double lo=-FunctionDefaults<NDIM>::get_cell_width()[0]*0.5*scale;

    const long nstep=150;
    const double stepsize=(hi-lo)/nstep;

    if(world.rank() == 0) {
      FILE *f =  0;
      f=fopen(filename.c_str(), "w");
      if(!f) MADNESS_EXCEPTION("plot_along: failed to open the plot file", 0);


      for (int i0=0; i0<nstep; i0++) {
        for (int i1=0; i1<nstep; i1++) {

          Vector<double,NDIM> coord(0.0);

          // plot y/z-plane
          coord[0]=lo+i0*stepsize;
          coord[2]=lo+i1*stepsize;

          fprintf(f,"%12.6f %12.6f %12.6f\n",coord[0],coord[2],function(coord));

        }
        // gnuplot-style
        fprintf(f,"\n");
      }
      fclose(f);
   }
}

void plot(const real_function_3d& f, const std::string filename, const long k) {
    FILE* file = fopen(filename.c_str(),"w");

    for (int i=0; i<k; ++i) {
    	double z=0.001+double(i)*0.01;
    	coord_3d r{0.0,0.0,z};
    	double c=f(r);
    	fprintf(file,"%lf %lf\n",z,c);
    }
    fclose(file);
}

// print the radial density: r^2 rho
void plot_radial_density(const real_function_3d& rho, const std::string filename, const tensorT& grid) {
	FILE* file = fopen(filename.c_str(),"w");
	for (int i=0; i<grid.dim(0); ++i) {
		double r=grid(i);
		coord_3d xyz{0.0,0.0,r};
		double c=r*r*rho(xyz);
		fprintf(file,"%lf %lf\n",r,c);
	}
	fclose(file);
}

// print the radial function given on a grid
void plot_radial_function(const real_function_3d& rho, const std::string filename) {
	int N = 2000;
	double rmax=5;
	double step = rmax/N;
	Tensor<double> grid(N);
	for(int i=0; i<N; i++)
	    grid(i) = i*step;
	FILE* file = fopen(filename.c_str(),"w");
	for (int i=0; i<grid.dim(0); ++i) {
		double r=grid(i);
		coord_3d xyz{0.0,0.0,r};
		double c=rho(xyz);
		fprintf(file,"%lf %lf\n", r,c);
	}
	fclose(file);
}


void compute_energy(World& world, const real_function_3d& psi, const real_function_3d& pot,
double& ke, double& pe) {

	double kinetic_energy = 0.0;
    for (int axis=0; axis<3; axis++) {
    	real_derivative_3d D = free_space_derivative<double,3>(world, axis);
    	real_function_3d dpsi = D(psi);
    	kinetic_energy += 0.5*inner(dpsi,dpsi);
    }
    ke=kinetic_energy;

    pe=inner(psi,pot*psi);
    if(world.rank() == 0) {
        printf("compute the ke, pe, te at time   %4.1fs %12.8f %12.8f %12.8f\n",
        		wall_time(),ke,pe,ke+pe);
    }
}

/// apply the Green's function on V*psi, update psi and the energy

/// @return the error norm of the orbital
double iterate(World& world, const real_function_3d& VV, real_function_3d& psi, double& eps) {

	const double thresh=FunctionDefaults<3>::get_thresh()*0.1;
	real_function_3d Vpsi = (VV*psi);
    Vpsi.scale(-2.0).truncate(thresh);

    real_convolution_3d op = BSHOperator3D(world, sqrt(-2*eps), 1.e-7, 1e-7);
    real_function_3d tmp=op(Vpsi).truncate(thresh);

    double norm = tmp.norm2();
    real_function_3d r = tmp-psi;
    double rnorm = r.norm2();
    double eps_new = eps - 0.5*inner(Vpsi,r)/(norm*norm);
    if (world.rank() == 0) {
        print("norm=",norm," eps=",eps," err(psi)=",rnorm," err(eps)=",eps_new-eps);
    }
    psi = tmp.scale(1.0/norm);
    if (eps_new<0.0) eps = eps_new;
    return rnorm;
}

/// orthogonalize orbital i against all other orbitals
void orthogonalize(std::vector<real_function_3d>& orbitals, const int ii) {
	MADNESS_ASSERT(size_t(ii)<orbitals.size());

	real_function_3d& phi=orbitals[ii];

	// loop over all other orbitals
	for (int i=0; i<ii; ++i) {
		const real_function_3d orbital=orbitals[i];
		double ovlp=inner(orbital,phi);
//		double norm=orbital.norm2();
//		phi-=(ovlp/norm/norm)*orbital;
		phi=phi-(ovlp)*orbital;

	}

	double n=phi.norm2();
	phi.scale(1.0/n);
}

/// solve the residual equations

/// @param[in]		potential  the effective potential
/// @param[in]		thresh     the threshold for the error in the orbitals
/// @param[in,out]	eps      guesses for the orbital energies
/// @param[in,out]	orbitals the first n roots of the equation
void solve(World& world, const real_function_3d& potential, const double thresh, tensorT& eps,
		std::vector<real_function_3d>& orbitals) {

	const long nroots=eps.size();
	print("solving for",nroots,"roots of the effective potential");

	// loop over all roots
	for (int i=0; i<nroots; ++i) {

		real_function_3d& phi=orbitals[i];
		double& eiger=eps(i);
		if (world.rank()==0) print("\nworking on orbital",i);

//		double ke=0.0,pe=0.0;
		double error=1e4;
		int ii=0;
		while (error>thresh and ii++<15) {
//			compute_energy(world,phi,potential,ke,pe);
			error=iterate(world,potential,phi,eiger);
			orthogonalize(orbitals,i);
		}
		//save_function(world,phi,"orbital"+stringify(i));
//		load_function(world,phi,"orbital"+stringify(i));
//		std::string name="orbital"+stringify(i);
//		draw_plane(world,phi,name);
//		Tensor<double> S=matrix_inner(world,orbitals,orbitals);
//		print("overlap");
//		print(S);

	}
}

/// given the density and the trial potential, update the potential to yield the density

/// @param[in]		world	the world
/// @param[in,out]	potential	the potential to be updated
/// @param[in]		density		the density
/// @param[in]		refdens		the reference density, or target densiy
void update_potential(World& world, real_function_3d& potential, const real_function_3d& density,
		const real_function_3d& refdens) {

	real_function_3d diff=(refdens-density).truncate();
//	diff=-0.01*(diff*hf.get_calc().vnuc);
	potential-=diff;
}

int main(int argc, char** argv) {
    initialize(argc, argv);
    { // limit lifetime of world so that finalize() can execute cleanly
      World world(SafeMPI::COMM_WORLD);
      //START_TIMER(world);

      try{
        // Load info for MADNESS numerical routines
        startup(world,argc,argv,true);
        print_meminfo(world.rank(), "startup");
        FunctionDefaults<3>::set_pmap(pmapT(new LevelPmap< Key<3> >(world)));

        std::cout.precision(6);

        // Process 0 reads input information and broadcasts
        const char * inpname = "input";
        for (int i=1; i<argc; i++) {
            if (argv[i][0] != '-') {
                inpname = argv[i];
                break;
            }
        }
        if (world.rank() == 0) print("input filename: ", inpname);
        if (!file_exists(inpname)) {
            throw "input file not found!";
        }
        SCF calc(world, inpname);

        // greetings
        if (world.rank() == 0) {
          print("\n\n");
          print(" MADNESS Hartree-Fock and Density Functional Theory Program");
          print(" ----------------------------------------------------------\n");
          print("\n");
          calc.molecule.print();
          print("\n");
          calc.param.print(world);
        }
        //END_TIMER(world, "initialize");

        // Come up with an initial OK data map
        if (world.size() > 1) {
          calc.set_protocol<3>(world,calc.param.econv);
          calc.make_nuclear_potential(world);
          calc.initial_load_bal(world);
        }
        calc.set_protocol<3>(world,calc.param.protocol_data[0]);

        MolecularEnergy E(world, calc);
        E.value(calc.molecule.get_all_coords().flat()); // ugh!

        real_function_3d rho = calc.make_density(world, calc.aocc, calc.amo);
        real_function_3d brho = rho;
        if (calc.param.nbeta != 0 && !calc.param.spin_restricted)
            brho = calc.make_density(world, calc.bocc, calc.bmo);
        rho.gaxpy(1.0, brho, 1.0);

/***********************************/
	const double thresh=FunctionDefaults<3>::get_thresh();
	print("global thresh = ", thresh);

	real_function_3d trial_pot=real_factory_3d(world).empty();

        int ispin = 0;
        XCOperator xcoperator(world,"LDA",ispin,rho,brho);
        trial_pot=xcoperator.make_xc_potential();

	functionT vcoul = apply(*(calc.coulop), rho);
        functionT vlocal;
	real_function_3d vnuc;
	vnuc = calc.potentialmanager->vnuclear();
        vlocal = vcoul + vnuc;

        // now we have the functions refdens and potential
        print("start oep iterations!!!!");
        const double n_el_ref=rho.trace();
        print("refdens.trace()    ",n_el_ref);

        // solve the residual equations
        std::vector<real_function_3d> orbitals=copy(world,calc.amo);
        tensorT eps=calc.aeps;
	int iter = 0;
	int max_iter = 100;
	bool cnvgd = false;
	double oep_thresh = thresh*5.0;
        while(!cnvgd && iter< max_iter) {
		iter++;
		functionT vtot;
		vtot = vlocal + trial_pot;
		vtot.truncate();

                // given the potential, solve for the first roots (the orbitals)
                solve(world,vtot,thresh,eps,orbitals);

                // make the density from the orbitals
                real_function_3d trial_dens=2.0*calc.make_density(world,calc.aocc,orbitals);

                // print out the status
                real_function_3d diffdens=trial_dens-rho;
                double error=(diffdens).norm2();
                if (world.rank()==0) print("error in the density: iteration, diff.norm2()",iter,error);
                //save_function(world,diffdens,prefix+"diffdens_iteration"+stringify(i));

		if(error < oep_thresh) cnvgd = true;
                // given the trial density and the target density, update the potential
                update_potential(world,trial_pot,trial_dens,rho);
//                save_function(world,potential,prefix+"potential_iteration"+stringify(i));
        }
	
	plot_radial_function(trial_pot, "radial_potential");

      }


      catch (const SafeMPI::Exception& e) {
        print(e);
        error("caught an MPI exception");
      }
      catch (const madness::MadnessException& e) {
        print(e);
        error("caught a MADNESS exception");
      }
      catch (const madness::TensorException& e) {
        print(e);
        error("caught a Tensor exception");
      }
      catch (const char* s) {
        print(s);
        error("caught a string exception");
      }
      catch (const std::string& s) {
        print(s);
        error("caught a string (class) exception");
      }
      catch (const std::exception& e) {
        print(e.what());
        error("caught an STL exception");
      }
      catch (...) {
        error("caught unhandled exception");
      }

      world.gop.fence();
      world.gop.fence();
      print_stats(world);
    }
    finalize();
    return 0;
}

//Xing

/// \file oep.h
/// \brief calculate xc potential from given density
/// \defgroup moldft The molecular density funcitonal and Hartree-Fock code

#ifndef MADNESS_CHEM_OEPMAN_H__INCLUDED
#define MADNESS_CHEM_OEPMAN_H__INCLUDED


#include <chem/SCF.h>
namespace madness {

    class OEPMAN {

	struct OEPparameters {
	       double oep_thresh;	   ///< thresh for wavelet expansion
	       double oepconv;             ///< OEP density convergence
	       double wconv;		   ///< Wu-Yang W convergence
	       int oep_max_iter;	   ///< max number of iteration
	       int oep_init; 		   ///< initil vxc type: 0, zero; 1, HF exchange 
	       int solver;		   ///< OEP solvers: 1, Wu-Yang; 2, new
	       bool readdens;		   ///< if true read density from file
	       int plot_NGrid[3];	   ///< plot grid number
	       double plot_box[6];	   ///< plot box coords


               OEPparameters(const std::string& input)
	       :oep_thresh(1e-6)
	       ,oepconv(1e-4)
	       ,wconv(1e-6)
	       ,oep_max_iter(50)
	       ,readdens(false)
	       ,solver(1)
	       ,oep_init(1)
	       {
                // get the parameters from the input file
                std::ifstream f(input.c_str());
                position_stream(f, "oepman");
                std::string s;

		for(int i=0; i<3; i++) plot_NGrid[i] = 1;
		for(int i=0; i<6; i++) plot_box[i] = 0.0;

                while (f >> s) {
                    if (s == "end") break;
                    else if (s == "oepconv") f >> oepconv;
		    else if (s == "wconv") f >> wconv;
		    else if (s == "readdens") readdens = true;
		    else if (s == "oep_thresh") f >> oep_thresh;
		    else if (s == "solver") f >> solver;
		    else if (s == "oep_init") f >> oep_init;
		    else if (s == "oep_max_iter") f >> oep_max_iter;
		    else if (s == "plot_box_x" || s == "plot_box_y" || s == "plot_box_z") {
			int xyz, i=0;
			if(s == "plot_box_x") xyz = 0;
			else if(s == "plot_box_y") xyz = 1;
			else if(s == "plot_box_z") xyz = 2;

			std::string buf;
                	std::getline(f,buf);
			double d;
                	std::stringstream s(buf);
                	while (s >> d) {
			  if(i==0) plot_NGrid[xyz] = (int)d;
			  else if(i==1) plot_box[2*xyz] = d;
			  else if(i==2) plot_box[2*xyz+1] = d;
			  i++;
			}
		    }
                    else {
			std::cout << "oep: unrecognized input keyword " << s << std::endl;
                	MADNESS_EXCEPTION("input error",0);
		    }
                }

               }
	};

	World& world;                           ///< the world
	SCF& calc;				///< SCF class
        OEPparameters param;                    ///< parameters for OEP
	vecfuncT refrho;			///< targrt density

    private:
	void greetings(World& world) {
	  if(world.rank() == 0){
	     print(" ----------------------------------------------------------");
	     print(" *                 Welcome to OEPMAN                      *");
	     print(" *                 Author: Xing Zhang                     *");
	     print(" *                      03/2017                           *");
             print(" ----------------------------------------------------------\n");
	  }
	  return;
	}
	void construct_refdens(World& world);
	void read_dens_from_file(){return;}

	void init_vxc_hfexch(functionT& vxc, functionT& vcoul);

	void plot_vxc(const functionT& vxc, const std::string filename);

	void update_potential_cg(World& world, int iter, double& W, const double Win,
                functionT& g_old, functionT& g,
                functionT& x_old, functionT& x,
                functionT& s_old, functionT& s,
                const functionT& vnuc, const functionT& vcoul, functionT& vxc, const functionT& refrho);

	double line_search_bt(World& world, double& W, const double Win, 
                                  const functionT& xin, const functionT& s, const functionT& g,
                                  const functionT& vnuc, const functionT& vcoul, const functionT& refrho);

    public:
	functionT solve(World& world);
	void solver_wy(World& world, functionT& vxc, functionT& vcoul, functionT& vnuc);
	void solver_new(World& world, functionT& vxc, functionT& vcoul, functionT& vnuc) {return;} //NYI
        void anal(World& world, functionT& vxc, const std::string& input);

	//constructor
	OEPMAN(World& world1, SCF& calc1, const std::string& input)
	:world(world1)
	,calc(calc1)
	,param(input)  //read params
	{
	  //say hello
	  greetings(world);
	  //get targrt density
	  construct_refdens(world);
	  //solve for V_xc
	  functionT vxc = solve(world);
	  //analize V_xc
	  anal(world, vxc, input);

	}

    };
}
#endif


#include "oepman.h"

namespace madness {

    double zero_func(const coordT& x) {return 0.0;}

    void OEPMAN::construct_refdens(World& world) {
	if(!param.readdens) { //refrho from SCF calculation
          refrho.push_back(calc.make_density(world, calc.aocc, calc.amo));
          if (calc.param.nbeta != 0 && !calc.param.spin_restricted)
              refrho.push_back(calc.make_density(world, calc.bocc, calc.bmo));
          else
	      refrho[0] *= 2.0; //total density at position 0
	}
	else {
	  read_dens_from_file(); //NYI
	}

	return;
    }

    functionT OEPMAN::solve(World& world) {
        //defualt zero vxc at beginning
	functionT vxc = functionT(factoryT(world).f(zero_func));

	double thresh=FunctionDefaults<3>::get_thresh();
	if(thresh != param.oep_thresh)
	  calc.set_protocol<3>(world, param.oep_thresh);

	functionT vcoul  = apply(*calc.coulop, refrho[0]);
        functionT vnuc   = calc.potentialmanager->vnuclear();

	//initialize vxc
	if(param.oep_init == 1) { //approximate hartree fock exchange
	  init_vxc_hfexch(vxc, vcoul);
	}

	//solve for vxc
	if(param.solver == 1) {
	  solver_wy(world, vxc, vcoul, vnuc);
	}
	else if(param.solver == 2) {
	  solver_new(world, vxc, vcoul, vnuc);
	}

	return vxc;
    }


    void OEPMAN::init_vxc_hfexch(functionT& vxc, functionT& vcoul) { //approximate HF exchange
        int n_elec = calc.param.nalpha + calc.param.nbeta;
        double factor = -1.0/(double)n_elec;
        vxc = vcoul*factor;

	return;
    }

    void OEPMAN::solver_wy(World& world, functionT& vxc, functionT& vcoul, functionT& vnuc) {
	int iter = 0;
        bool cnvgd = false;

	double w_old, w;
	functionT x_old, x, g_old, g, s_old, s;
	x = vxc;
	while(!cnvgd && iter< param.oep_max_iter) {
           iter++;

           if(iter == 1)
             w = calc.solve(world, vnuc, vcoul, x, refrho[0]);

           // make the density from the orbitals
           functionT rho=2.0*calc.make_density(world,calc.aocc,calc.amo);

           // print out the status
           functionT g = refrho[0] - rho;
           double error=(g).norm2();
           if (world.rank()==0) print(" Error in the density: iteration, diff.norm2()",iter,error);

           if(error < param.oepconv || fabs(w-w_old) < param.wconv ){
              cnvgd = true;
              vxc = x;
           }
           // given the trial density and the target density, update the potential
           g.truncate();
           if (world.rank()==0) print(" Wu-Yang W = ",w);
           if(cnvgd) break;
           w_old = w;
           update_potential_cg(world,iter,w,w_old,g_old,g,x_old,x,s_old,s, vnuc, vcoul, x, refrho[0]);

           x_old = x;
           g_old = g;
           s_old = s;
        }

	return;
    }



    void OEPMAN::update_potential_cg(World& world, int iter, double& W, const double Win,
                functionT& g_old, functionT& g,
                functionT& x_old, functionT& x,
                functionT& s_old, functionT& s,
                const functionT& vnuc, const functionT& vcoul, functionT& vxc, const functionT& refrho)
    {
        double alpha;

        if(iter == 1){
           s = -1.0 * g;
           alpha = line_search_bt(world, W, Win, x, s, g, vnuc, vcoul, refrho);
           x += alpha * s;
           return;
        }

        functionT gdiff = g-g_old;
        double beta1 = inner(g, gdiff)/inner(g_old, g_old);
        double beta = std::max(0.0, beta1);
        s = beta * s_old - g;
        s.truncate();

        alpha = line_search_bt(world, W, Win, x, s, g, vnuc, vcoul, refrho);
        x += alpha * s;

        return;
    }


    double OEPMAN::line_search_bt(World& world, double& W, const double Win, 
				  const functionT& xin, const functionT& s, const functionT& g,
                       		  const functionT& vnuc, const functionT& vcoul, const functionT& refrho)
    { //back tracking
        double alpha = 1.0;
        double sigma = 0.1;
        double factor = sigma * inner(g,s);
        functionT x;
        while(1){
           x = xin + alpha * s;
           W = calc.solve(world, vnuc, vcoul, x, refrho);

           if(W < Win + alpha * factor ) return alpha;
           else{
                alpha *= 0.5;
           }

        }
        return alpha;
    }



    void OEPMAN::anal(World& world, functionT& vxc, const std::string& input) {
	
	plot_vxc(vxc, input+".oep_vxc");

	return;
    }

    void OEPMAN::plot_vxc(const functionT& vxc, const std::string filename) {
        double step[3];
	for(int i=0; i<3; i++) {
	   step[i] = 0.0;
	   if(param.plot_NGrid[i] > 1)
	      step[i] = (param.plot_box[i*2+1] - param.plot_box[i*2])/(param.plot_NGrid[i]-1);
	}

        FILE* file = fopen(filename.c_str(),"w");

	for(int x=0; x<param.plot_NGrid[0]; x++)
	for(int y=0; y<param.plot_NGrid[1]; y++)
        for(int z=0; z<param.plot_NGrid[2]; z++) {
            double xx = param.plot_box[0] + step[0] * x;
	    double yy = param.plot_box[2] + step[1] * y;
	    double zz = param.plot_box[4] + step[2] * z;

            coord_3d xyz{xx,yy,zz};
            double c=vxc(xyz);
            fprintf(file,"%lf %lf %lf %lf\n", xx,yy,zz,c);
        }
        fclose(file);
    }

}

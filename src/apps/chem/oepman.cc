
#include "oepman.h"
#include <madness/world/worldmem.h>

namespace madness {

    //double zero_func(const coordT& x) {return 0.0;}

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
	functionT vxc = functionT(factoryT(world).fence(false).compressed(true).initial_level(1));

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
	  //calc.solve(world, vnuc, vcoul, vxc, refrho[0]);
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

           if(error < param.oepconv) { // || fabs(w-w_old) < param.wconv ){
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
	bool reset = false;

        if(iter == 1){
           s = -1.0 * g;
           alpha = line_search_bt(world, W, Win, reset, x, s, g, vnuc, vcoul, refrho);
           x += alpha * s;
           return;
        }

        functionT gdiff = g-g_old;
        double beta1 = inner(g, gdiff)/inner(g_old, g_old);
        double beta = std::max(0.0, beta1);
        s = beta * s_old - g;
        s.truncate();

        alpha = line_search_bt(world, W, Win, reset, x, s, g, vnuc, vcoul, refrho);
	if(reset){ //reset direction
	   if(world.rank() == 0) print(" reseting opt direction");
	   alpha = 0.95;
	   s = -1.0 * g;
	   alpha = line_search_bt(world, W, Win, reset, x, s, g, vnuc, vcoul, refrho);
	   if(reset) MADNESS_EXCEPTION(" line search failed",0); 
	   if(world.rank() == 0) print(" line search alpha = ", alpha);
	   x += alpha * s;
	   return;
	}
	if(world.rank() == 0) print(" line search alpha = ", alpha);
        x += alpha * s;

        return;
    }


    double OEPMAN::line_search_bt(World& world, double& W, const double Win, bool& reset, 
				  const functionT& xin, const functionT& s, const functionT& g,
                       		  const functionT& vnuc, const functionT& vcoul, const functionT& refrho)
    { //back tracking
        static double alpha = 1.0;
	static int count = 0;

	if(count > 0){
	  if(alpha >= 1.0)
	     alpha *= 1.5; //increase
	  else
	     alpha = 1.0;  //reset
	}

        double sigma = 0.1;
        double factor = sigma * inner(g,s);
        functionT x;

	count++;
	int count_line = 0;
	reset = false;
        while(count_line < 20){
	   count_line++;
           x = xin + alpha * s;
           W = calc.solve(world, vnuc, vcoul, x, refrho);

           if(W < Win + alpha * factor ) return alpha;
           else{
                alpha *= 0.5;
           }
        }
	if(count_line >= 20) reset = true; 
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


    //constrained optimization with KAIN
    void OEPMAN::solver_new(World& world, functionT& vxc, functionT& vcoul, functionT& vnuc) {
	functionT arho_old, brho_old;
        const double dconv = std::max(FunctionDefaults < 3 > ::get_thresh(),
                                      param.oepconv);
        const double trantol = calc.vtol / std::min(30.0, double(calc.amo.size()));
        const double tolloc = std::min(1e-6,0.01*dconv);
        double update_residual = 0.0, bsh_residual = 0.0;
        subspaceT subspace;
        tensorT Q;
        bool do_this_iter = true;
        bool converged = false;

	int maxsub_save = param.maxsub;
        param.maxsub = 2;

        for (int iter = 0; iter < param.oep_max_iter; ++iter) {
            if (world.rank() == 0)
                printf("\nIteration %d at time %.1fs\n\n", iter, wall_time());

	    if (iter > 0 && update_residual < 0.1) {
                //do_this_iter = false;
                param.maxsub = maxsub_save;
            }

            if (calc.param.localize && do_this_iter) {
                distmatT dUT;
                if (calc.param.localize_pm)
                  dUT = calc.localize_PM(world, calc.amo, calc.aset, tolloc, 0.1, iter == 0, true);
                else
                  dUT = calc.localize_boys(world, calc.amo, calc.aset, tolloc, 0.1, iter == 0);

                dUT.data().screen(trantol);
                calc.amo = transform(world, calc.amo, dUT);
                truncate(world, calc.amo);
                normalize(world, calc.amo);
                if (!calc.param.spin_restricted && calc.param.nbeta != 0) {
                  if (calc.param.localize_pm)
                    dUT = calc.localize_PM(world, calc.bmo, calc.bset, tolloc, 0.1, iter == 0, true);
                  else
                    dUT = calc.localize_boys(world, calc.bmo, calc.bset, tolloc, 0.1, iter == 0);

                    dUT.data().screen(trantol);
                    calc.bmo = transform(world, calc.bmo, dUT);
                    truncate(world, calc.bmo);
                    normalize(world, calc.bmo);
                }
            }


            functionT arho = calc.make_density(world, calc.aocc, calc.amo), brho;
            
            if (calc.param.nbeta) {
                if (calc.param.spin_restricted) {
                    brho = arho;
                } else {
                    brho = calc.make_density(world, calc.bocc, calc.bmo);
                }
            } else {
                brho = functionT(world); // zero
            }
            print_meminfo(world.rank(), "Make densities");
            
            if (iter < 2 || (iter % 10) == 0) {
                calc.loadbal(world, arho, brho, arho_old, brho_old, subspace);
                print_meminfo(world.rank(), "Load balancing");
            }
            double da = 0.0, db = 0.0;
            if (iter > 0) {
                da = (arho - arho_old).norm2();
                db = (brho - brho_old).norm2();
                if (world.rank() == 0)
                    print("delta rho", da, db, "residuals", bsh_residual,
                          update_residual);
                
            }
            
            arho_old = arho;
            brho_old = brho;
            functionT rho = arho + brho;
            rho.truncate();
            double enuclear = inner(rho, vnuc);

            functionT vlocal;
            double ecoulomb = 0.5 * inner(rho, vcoul);
	    double exca = 0.0, excb = 0.0;
            exca = 0.5*inner(rho, vxc); //approximate xc energy

	    functionT diffrho = rho - refrho[0];
	    double error = diffrho.norm2();
	    diffrho.clear(false);
	    if (world.rank()==0) print(" Error in the density: iteration, diff.norm2()",iter,error);
            rho.clear(false);

            vlocal = vcoul + vnuc + vxc;
            vlocal.truncate();

	    vecfuncT Vpsia, Vpsib;
            Vpsia = mul_sparse(world, vlocal, calc.amo, calc.vtol);
            truncate(world, Vpsia);
            world.gop.fence();
            if (!calc.param.spin_restricted && calc.param.nbeta) {
                Vpsib = mul_sparse(world, vlocal, calc.bmo, calc.vtol);;
                truncate(world, Vpsib);
                world.gop.fence();
            }

            double ekina = 0.0, ekinb = 0.0;
            tensorT focka = calc.make_fock_matrix(world, calc.amo, Vpsia, calc.aocc, ekina);
            tensorT fockb = focka;
            
            if (!calc.param.spin_restricted && calc.param.nbeta != 0)
                fockb = calc.make_fock_matrix(world, calc.bmo, Vpsib, calc.bocc, ekinb);
            else if (calc.param.nbeta != 0) {
                ekinb = ekina;
            }
            
            if (!calc.param.localize && do_this_iter) {
                tensorT U = calc.diag_fock_matrix(world, focka, calc.amo, Vpsia, calc.aeps, calc.aocc,
                                              FunctionDefaults < 3 > ::get_thresh());
                //rotate_subspace(world, U, subspace, 0, amo.size(), trantol); ??
                if (!calc.param.spin_restricted && calc.param.nbeta != 0) {
                    U = calc.diag_fock_matrix(world, fockb, calc.bmo, Vpsib, calc.beps, calc.bocc,
                                              FunctionDefaults < 3 > ::get_thresh());
                    //rotate_subspace(world, U, subspace, amo.size(), bmo.size(),trantol);
                }
            }
            
            double enrep = calc.molecule.nuclear_repulsion_energy();
            double ekinetic = ekina + ekinb;
            double exc = exca + excb;
            double etot = ekinetic + enuclear + ecoulomb + exc + enrep;
            //current_energy = etot;
            
            if (world.rank() == 0) {
                //lots of dps for testing Exc stuff
                /*printf("\n              kinetic %32.24f\n", ekinetic);
                printf("         nonlocal psp %32.24f\n", enonlocal);
                printf("   nuclear attraction %32.24f\n", enuclear);
                printf("              coulomb %32.24f\n", ecoulomb);
                printf(" exchange-correlation %32.24f\n", exc);
                printf("    nuclear-repulsion %32.24f\n", enrep);
                printf("                total %32.24f\n\n", etot);*/

                printf("\n              kinetic %16.8f\n", ekinetic);
                printf("   nuclear attraction %16.8f\n", enuclear);
                printf("              coulomb %16.8f\n", ecoulomb);
                printf(" exchange-correlation %16.8f\n", exc);
                printf("    nuclear-repulsion %16.8f\n", enrep);
                printf("                total %20.12f\n\n", etot);
            }
            
            if (iter > 0) {
                if (da < dconv * calc.molecule.natom() && db < dconv * calc.molecule.natom()
                    && bsh_residual < dconv) converged=true;

                // do diagonalization etc if this is the last iteration, even if the calculation didn't converge
                if (converged || iter==param.oep_max_iter-1) {
                    if (world.rank() == 0 && converged) {
                        print("\nConverged!\n");
                    }
                    
                    // Diagonalize to get the eigenvalues and if desired the final eigenvectors
                    tensorT U;
                    tensorT overlap = matrix_inner(world, calc.amo, calc.amo, true);
		    print("overlap matrix:");
		    print(overlap);
		    print("fock matrix:");              
		    print(focka);      

                    sygvp(world, focka, overlap, 1, U, calc.aeps);
                    
                    if (!calc.param.localize) {
                        calc.amo = transform(world, calc.amo, U, trantol, true);
                        truncate(world, calc.amo);
                        normalize(world, calc.amo);
                    }
                    if (calc.param.nbeta != 0 && !calc.param.spin_restricted) {

                        overlap = matrix_inner(world, calc.bmo, calc.bmo, true);
                        
                        sygvp(world, fockb, overlap, 1, U, calc.beps);
                        
                        if (!calc.param.localize) {
                            calc.bmo = transform(world, calc.bmo, U, trantol, true);
                            truncate(world, calc.bmo);
                            normalize(world, calc.bmo);
                        }
                    }
                    
                    if (world.rank() == 0) {
                        print(" ");
                        print("alpha eigenvalues");
                        print (calc.aeps);
                        if (calc.param.nbeta != 0 && !calc.param.spin_restricted) {
                            print("beta eigenvalues");
                            print (calc.beps);
                        }


                        // write eigenvalues etc to a file at the same time for plotting DOS etc.
                        FILE *f=0;
                        if (calc.param.nbeta != 0 && !calc.param.spin_restricted) {
                            f = fopen("energies_alpha.dat", "w");}
                        else{
                            f = fopen("energies.dat", "w");}

                        long nmo = calc.amo.size();
                        fprintf(f, "# %8li\n", nmo);
                        for (long i = 0; i < nmo; ++i) {
                            fprintf(f, "%13.8f\n", calc.aeps(i));
                        }
                        fclose(f);

                        if (calc.param.nbeta != 0 && !calc.param.spin_restricted) {
                            long nmo = calc.bmo.size();
                            FILE *f=0;
                            f = fopen("energies_beta.dat", "w");

                            fprintf(f, "# %8li\n", nmo);
                            for (long i = 0; i < nmo; ++i) {
                                fprintf(f, "%13.8f\t", calc.beps(i));
                            }
                            fclose(f);
                        }

                    }
                    
                    if (calc.param.localize) {
                        // Restore the diagonal elements for the analysis
                        for (unsigned int i = 0; i < calc.amo.size(); ++i)
                            calc.aeps[i] = focka(i, i);
			print("debug:");
                        print("alpha eigenvalues for localized orbitals");
                        print (calc.aeps);
                        if (calc.param.nbeta != 0 && !calc.param.spin_restricted)
                            for (unsigned int i = 0; i < calc.bmo.size(); ++i)
                                calc.beps[i] = fockb(i, i);
                    }
                    
                    break;
                }
                
            }
            
            update_subspace(world, vxc, Vpsia, Vpsib, focka, fockb, subspace, Q,
                            bsh_residual, update_residual);

        }




	return;
    }




    void OEPMAN::update_subspace(World & world, functionT& vxc, vecfuncT & Vpsia, vecfuncT & Vpsib,
                              tensorT & focka, tensorT & fockb, subspaceT & subspace, tensorT & Q,
                              double & bsh_residual, double & update_residual) {
        double aerr = 0.0, berr = 0.0;
        vecfuncT vm = calc.amo;
	vm.push_back(vxc);
        
        // Orbitals with occ!=1.0 exactly must be solved for as eigenfunctions
        // so zero out off diagonal lagrange multipliers
        for (int i = 0; i < calc.param.nmo_alpha; i++) {
            if (calc.aocc[i] != 1.0) {
                double tmp = focka(i, i);
                focka(i, _) = 0.0;
                focka(_, i) = 0.0;
                focka(i, i) = tmp;
            }
        }
  
        vecfuncT rm = calc.compute_residual(world, calc.aocc, focka, calc.amo, Vpsia, aerr);
	functionT rho = calc.make_density(world, calc.aocc, calc.amo);
	rho *= 2.0;
	functionT diffrho = refrho[0] - rho;
	rho.clear(false);
        //diffrho = diffrho/rho;
        diffrho.truncate();

	rm.push_back(diffrho);
	double rho_err = diffrho.norm2();
	aerr = std::max(aerr, rho_err);
/*	NYI
        if (calc.param.nbeta != 0 && !calc.param.spin_restricted) {
            for (int i = 0; i < calc.param.nmo_beta; i++) {
                if (calc.bocc[i] != 1.0) {
                    double tmp = fockb(i, i);
                    fockb(i, _) = 0.0;
                    fockb(_, i) = 0.0;
                    fockb(i, i) = tmp;
                }
            }
            
            vecfuncT br = calc.compute_residual(world, calc.bocc, fockb, calc.bmo, Vpsib, berr);
            vm.insert(vm.end(), bmo.begin(), bmo.end());
            rm.insert(rm.end(), br.begin(), br.end());
        }
*/
        bsh_residual = std::max(aerr, berr);
        world.gop.broadcast(bsh_residual, 0);
        compress(world, vm, false);
        compress(world, rm, false);
        world.gop.fence();
        subspace.push_back(pairvecfuncT(vm, rm));
        int m = subspace.size();
        tensorT ms(m);
        tensorT sm(m);
        for (int s = 0; s < m; ++s) {
            const vecfuncT & vs = subspace[s].first;
            const vecfuncT & rs = subspace[s].second;
            for (unsigned int i = 0; i < vm.size(); ++i) {
                ms[s] += vm[i].inner_local(rs[i]);
                sm[s] += vs[i].inner_local(rm[i]);
            }
        }
        
        world.gop.sum(ms.ptr(), m);
        world.gop.sum(sm.ptr(), m);
        tensorT newQ(m, m);
        if (m > 1)
            newQ(Slice(0, -2), Slice(0, -2)) = Q;
        
        newQ(m - 1, _) = ms;
        newQ(_, m - 1) = sm;
        Q = newQ;
        if (world.rank() == 0) { print("kain Q"); print(Q); }
        tensorT c;
        if (world.rank() == 0) {
            double rcond = 1e-12;
            while (1) {
                c = KAIN(Q, rcond);
                if (world.rank() == 0) print("kain c:", c);
                if (std::abs(c[m - 1]) < 3.0) {
                    break;
                } else if (rcond < 0.01) {
                    print("Increasing subspace singular value threshold ", c[m - 1],
                          rcond);
                    rcond *= 100;
                } else {
                    print("Forcing full step due to subspace malfunction");
                    c = 0.0;
                    c[m - 1] = 1.0;
                    break;
                }
            }
        }

        world.gop.broadcast_serializable(c, 0);
        if (world.rank() == 0) {
            print("Subspace solution", c);
        }
        vecfuncT amo_new = zero_functions_compressed<double, 3>(world, calc.amo.size(), false);
	functionT vxc_new = functionT(factoryT(world).fence(false).compressed(true).initial_level(1));
//        vecfuncT bmo_new = zero_functions_compressed<double, 3>(world, calc.bmo.size(), false);
        world.gop.fence();
        for (unsigned int m = 0; m < subspace.size(); ++m) {
            const vecfuncT & vm = subspace[m].first;
            const vecfuncT & rm = subspace[m].second;
            const vecfuncT vma(vm.begin(), vm.begin() + calc.amo.size());
	    const functionT vma_last = vm[vm.size()-1];
            const vecfuncT rma(rm.begin(), rm.begin() + calc.amo.size());
	    const functionT rma_last = rm[rm.size()-1];
//            const vecfuncT vmb(vm.end() - bmo.size(), vm.end());
//            const vecfuncT rmb(rm.end() - bmo.size(), rm.end());
            gaxpy(world, 1.0, amo_new, c(m), vma, false);
	    vxc_new += c(m) * vma_last; 
            gaxpy(world, 1.0, amo_new, -c(m), rma, false);
	    vxc_new -= c(m) * rma_last;
//            gaxpy(world, 1.0, bmo_new, c(m), vmb, false);
//            gaxpy(world, 1.0, bmo_new, -c(m), rmb, false);
        }
        world.gop.fence();
        if (param.maxsub <= 1) {
            subspace.clear();
        } else if (subspace.size() == param.maxsub) {
            subspace.erase(subspace.begin());
            Q = Q(Slice(1, -1), Slice(1, -1));
        }
        
        do_step_restriction(world, calc.amo, amo_new);
        calc.orthonormalize(world, amo_new, calc.param.nalpha);
        calc.amo = amo_new;

        do_step_restriction_func(world, vxc, vxc_new);
	vxc = vxc_new;
        
        if (!calc.param.spin_restricted && calc.param.nbeta != 0) {
//            do_step_restriction(world, bmo, bmo_new, "beta");
//            orthonormalize(world, bmo_new, param.nbeta);
//            bmo = bmo_new;
        } else {
            calc.bmo = calc.amo;
        }
    }



    double OEPMAN::do_step_restriction(World& world, const vecfuncT& mo, vecfuncT& mo_new) const {
        std::vector<double> anorm = norm2s(world, sub(world, mo, mo_new));
        int nres = 0;
        for (unsigned int i = 0; i < mo.size(); ++i) {
            if (anorm[i] > param.maxrotn) {
                double s = param.maxrotn / anorm[i];
                ++nres;
                if (world.rank() == 0) {
                    if (nres == 1)
                        printf("  restricting step for orbitals:");
                    printf(" %d", i);
                }
                mo_new[i].gaxpy(s, mo[i], 1.0 - s, false);
            }
        }
        if (nres > 0 && world.rank() == 0)
            printf("\n");

        world.gop.fence();
        double rms, maxval;
        calc.vector_stats(anorm, rms, maxval);
        if (world.rank() == 0)
            print("Norm of vector changes", ": rms", rms, "   max", maxval);
        return maxval;
    }

    double OEPMAN::do_step_restriction_func(World& world, const functionT& vxc, functionT& vxc_new) const {
	functionT diff = vxc - vxc_new;
        double anorm = diff.norm2();
	diff.clear(false);
        if (anorm > param.maxrotn) {
                double s = param.maxrotn / anorm;
                if (world.rank() == 0) {
                    printf("  restricting step for potential:");
                }
                vxc_new.gaxpy(s, vxc, 1.0 - s, false);
        }

        world.gop.fence();
        if (world.rank() == 0)
            print("Norm of Vxc changes", anorm);
        return anorm;
    }

}

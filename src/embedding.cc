/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libfock/jk.h"
#include "math.h" 
#include <string>
#include "embedding.h"
#include "orbital-helper/iao_builder.h"

namespace psi {
namespace forte {

//using namespace ambit;

Embedding::Embedding(SharedWavefunction ref_wfn, Options& options, 
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info)
    :Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {

    shallow_copy(ref_wfn);
    ref_wfn_ = ref_wfn;
    print_method_banner({"Embedded calculation with IAO localization","Nan He"});

    startup();
}

void Embedding::startup() {
}

SharedMatrix AO_TEI_all(SharedWavefunction wfn) {
	MintsHelper mints(wfn);
	SharedMatrix TEI = mints.ao_eri();
	return TEI;
}

double AO_TEI_single(SharedWavefunction wfn, int u, int v, int l, int s, Dimension nmopi) {
	MintsHelper mints(wfn);
	SharedMatrix TEI = mints.ao_eri();
	double eri_value = TEI->get(u*nmopi[0] + v, l*nmopi[0] + s); //Currently works only for C1 !!!
	return eri_value;
}

double eri_index(SharedMatrix TEI, int u, int v, int l, int s, Dimension nmopi) {
	return TEI->get(u*nmopi[0] + v, l*nmopi[0] + s); //Currently works only for C1 !!!
}

void build_D(SharedMatrix C, Dimension noccpi, SharedMatrix D) {
	int nirrep = C->nirrep();
	Dimension nmopi = C->colspi();
	for (int h = 0; h < nirrep; ++h) {
		for (int u = 0; u < nmopi[h]; ++u) {
			for (int v = 0; v < nmopi[h]; ++v) {
				double tmp = 0.0;
				for (int m = 0; m < noccpi[h]; ++m) {
					tmp += C->get(h, u, m)*C->get(h, v, m);
				}
				D->set(h, u, v, tmp);
			}
		}
	}
}

double calculate_HF_energy(SharedMatrix hpF, SharedMatrix D) {
	//hpF = h_core + Fock_current
	int nirrep = D->nirrep();
	Dimension nmopi = D->colspi();
	double E = 0.0;
	for (int h = 0; h < nirrep; ++h) {
		for (int u = 0; u < nmopi[h]; ++u) {
			for (int v = 0; v < nmopi[h]; ++v) {
				E += D->get(h, u, v) * hpF->get(h, u, v);
			}
		}
	}
	return E;
}

SharedMatrix build_Cocc(SharedMatrix C, int nirrep, Dimension nmopi, Dimension noccpi) {
	SharedMatrix C_occ(new Matrix("C_occ", nirrep, nmopi, nmopi)); //make it ready for changes!!here
	for (int h = 0; h < nirrep; ++h) {
		for (int u = 0; u < nmopi[h]; ++u) {
			for (int i = 0; i < noccpi[h]; ++i) {
				C_occ->set(h, u, i, C->get(h, u, i));
			}
		}
	}
	return C_occ;
}

SharedMatrix build_Focc(SharedMatrix F, int nirrep, Dimension nmopi, Dimension noccpi) {
	SharedMatrix F_occ(new Matrix("F_occ", nirrep, nmopi, nmopi)); //make it ready for changes!!here
	for (int h = 0; h < nirrep; ++h) {
		for (int u = 0; u < nmopi[h]; ++u) {
			for (int i = 0; i < noccpi[h]; ++i) {
				F_occ->set(h, u, i, F->get(h, u, i));
			}
		}
	}
	return F_occ;
}

void build_G(SharedWavefunction wfn, SharedMatrix C, SharedMatrix G, Options& options, int methods) {
	int nirrep = wfn->nirrep();
	Dimension nmopi = wfn->nmopi();
	Dimension noccpi = wfn->doccpi();

	if (methods == 0) {
		std::shared_ptr<JK> JK_occ;
		JK_occ = JK::build_JK(wfn->basisset(), wfn->get_basisset("DF_BASIS_SCF"), options);
		JK_occ->set_memory(Process::environment.get_memory() * 0.8);

		// JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
		//JK_occ->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
		JK_occ->initialize();
		JK_occ->set_do_J(true);
		// JK_core->set_allow_desymmetrization(true);
		JK_occ->set_do_K(true);

		SharedMatrix C_occ = build_Cocc(C, nirrep, nmopi, noccpi);

		std::vector<std::shared_ptr<Matrix>>& Cl = JK_occ->C_left();
		std::vector<std::shared_ptr<Matrix>>& Cr = JK_occ->C_right();
		Cl.clear();
		Cr.clear();
		Cl.push_back(C_occ);
		Cr.push_back(C_occ);

		JK_occ->compute();

		G->copy(JK_occ->J()[0]);
		//SharedMatrix G = JK_occ->J()[0];
		SharedMatrix K = JK_occ->K()[0];

		JK_occ->finalize();

		G->scale(2.0);
		G->subtract(K); //G = (2J-K)D

		//G->transform(C);
	}
	if (methods == 1) {
		SharedMatrix TEI = AO_TEI_all(wfn);
		SharedMatrix D(new Matrix("Density", nirrep, nmopi, nmopi));
		build_D(C, noccpi, D);
		G->zero();
		double tmp = 0.0;
		double jkfactor = 0.0;
		for (int h = 0; h < nirrep; ++h) {//This part doesn't work for symmetry higher than C1!!
			for (int u = 0; u < nmopi[h]; ++u) {
				for (int v = 0; v < nmopi[h]; ++v) {
					tmp = 0.0;
					for (int l = 0; l < nmopi[h]; ++l) {
						for (int s = 0; s < nmopi[h]; ++s) {
							jkfactor = 2 * eri_index(TEI, u, v, l, s, nmopi) - eri_index(TEI, u, l, v, s, nmopi); 
							tmp += D->get(h, l, s)*jkfactor;
						}
					}
					G->set(h, u, v, tmp);
				}
			}
		}
	}
}

double do_HF(SharedWavefunction wfn, SharedMatrix hcore, int nirrep, Dimension nmopi,
	Dimension noccpi, int Maxcyc, double E_conv, Options& options) { 

	//Hartree-Fock calculation with given core hamiltonian hcore

	SharedMatrix Fock(new Matrix("Fock", nirrep, nmopi, nmopi)); //AO Fock
	SharedMatrix S = wfn->S(); //overlap matrix
	SharedMatrix L(new Matrix("L", nirrep, nmopi, nmopi)); //S eigenvectors
	SharedVector lm(new Vector("lambda", nirrep, nmopi)); //S eigenvalues
	SharedVector lminvhalf(new Vector("lambda inv half", nirrep, nmopi)); //S eigenvalues -1/2
	SharedMatrix LM(new Matrix("LM", nirrep, nmopi, nmopi));

	//HF calculation for environment
	outfile->Printf("\n \n Hartree-Fock SCF calculation for given h_core in embedding \n");

	Fock->copy(hcore);

	//Construct -1/2 overlap matrix
	S->diagonalize(L, lm);

	for (int h = 0; h < nirrep; ++h) {
		for (int i = 0; i < nmopi[h]; ++i) {
			double tmp = 1.0 / lm->get(h, i);
			lminvhalf->set(h, i, sqrt(tmp));
		}
	}
	LM->set_diagonal(lminvhalf);

	SharedMatrix S_half = Matrix::triplet(L, LM, L, false, false, true);

	int count = 0;
	double eps = 0.0;
	double E_scf = 0.0;
	double Etmp = 0.0;
	SharedMatrix C_star(new Matrix("C*", nirrep, nmopi, nmopi));
	SharedVector epis(new Vector("Eigens", nirrep, nmopi));
	SharedMatrix Fock_uv(new Matrix("Fock uv", nirrep, nmopi, nmopi));
	SharedMatrix C_iter(new Matrix("C_iterate", nirrep, nmopi, nmopi));
	SharedMatrix hpF(new Matrix("hcore uv + F uv", nirrep, nmopi, nmopi));
	SharedMatrix Temp(new Matrix("Temperate Mat", nirrep, nmopi, nmopi));
	SharedMatrix D_iter(new Matrix("Density Matrix", nirrep, nmopi, nmopi));
	SharedMatrix G(new Matrix("G_uv", nirrep, nmopi, nmopi));

	while (count < Maxcyc) { //SCF loop
		//rotate F
		Fock_uv->copy(Fock);

		Temp->zero();
		Temp = Matrix::triplet(S_half, Fock, S_half, true, false, false);
		Fock->copy(Temp); //F' = S-1/2T F S-1/2

		//outfile->Printf("\n ****** Checkpoint: Fock_iter after rotation iter %d ******", count);
		//Fock->print();

		C_star->zero();
		epis->zero();
		Fock->diagonalize(C_star, epis);

		if (count == 0) {
			E_scf = epis->get(0);
		}

		//rotate C
		C_iter->zero();
		C_iter = Matrix::doublet(S_half, C_star, false, false);

		//calculate energy here
		build_D(C_iter, noccpi, D_iter);
		Etmp = E_scf;

		hpF->zero();
		hpF->add(hcore);
		hpF->add(Fock_uv);
		E_scf = calculate_HF_energy(hpF, D_iter);
		outfile->Printf("\n SCF iterations %d, E_scf = %8.8f", count, E_scf);
		eps = fabs(E_scf - Etmp);
		if (eps < E_conv) {
			outfile->Printf("\n SCF converged after %d iterations.", count);
			E_scf += wfn->molecule()->nuclear_repulsion_energy(wfn->get_dipole_field_strength());
			outfile->Printf("\n E_SCF_electron = %8.8f \n", E_scf);
			break;
		}

		//update Fock
		build_G(wfn, C_iter, G, options, 0);

		Fock->zero();

		Fock->add(hcore);
		Fock->add(G);
		++count;
	}
	return E_scf; //Change return values to Maps later!!!!
}

SharedMatrix Separate_orbital_subset(SharedWavefunction wfn, int natom_sys, std::map<std::string, SharedMatrix> Loc, int methods) {
	//Write separation code here soon
	outfile->Printf("\n *** Orbital assignment and separation *** \n");
	Dimension nmopi = wfn->nmopi();
	int nirrep = wfn->nirrep();
	std::shared_ptr<BasisSet> basis = wfn->basisset();

	//generate basis_sys for mol_sys later if we want different basis for sys and env!
	int nbf = basis->nbf();
	outfile->Printf("\n number of basis on all atoms: %d", nbf);
	//count how many basis on system atoms
	int count_basis = 0;
	for (int A = 0; A < natom_sys; A++) {
		int n_shell = basis->nshell_on_center(A);
		for (int Q = 0; Q < n_shell; Q++) {
			const GaussianShell& shell = basis->shell(A, Q);
			count_basis += shell.nfunction();
		}
	}
	outfile->Printf("\n number of basis on \"system\" atoms: %d", count_basis);

	//SharedMatrix C_sys(new Matrix("System Coeffs", nirrep, nbasispi, nbasispi));

	std::vector<int> index_trace = {};
	int count_orbs = 0;

	if (methods == 0) { //Simple method, assign orbitals according to largest C position
		//C can only be C1 due to IAO code, so ignore symmetry!
		outfile->Printf("\n Simple catagorization algorithm: largest C^2 \n");
		for (int i = 0; i < nmopi[0]; ++i) {
			outfile->Printf("\n Running over the %d th MO \n", i);
			//loop over C rows (i:nmopi[h]), find the largest value, return index
			SharedVector r = Loc["L_local"]->get_column(0, i);
			double tmp = 0.0;
			int index_this = 0;
			for (int j = 0; j < nmopi[0]; ++j) {
				if (fabs(r->get(j)) > tmp) {
					index_this = j;  //find the largest C index, record row number (AO index!)
					tmp = r->get(0, j);
				}
			}
			if (index_this < count_basis) {  //Change this if symmetry beyond c1 is allowed in the future
				//r belongs to system!
				index_trace.push_back(i); //record index of MO column
				outfile->Printf("\n find %d", i);
				++count_orbs;
			}
			//r belongs to environment, do nothing
		}
	}

	if (methods == 1) { // Milliken charge method based on partial charge
						//C can only be C1 due to IAO code, so ignore symmetry!
		outfile->Printf("\n Charge-based catagorization algorithm \n");
		for (int i = 0; i < nmopi[0]; ++i) {
			outfile->Printf("\n Running over the %d th MO \n", i);
			//loop over C rows (i:nmopi[h]), find the largest value, return index
			SharedVector r = Loc["Q"]->get_column(0, i);
			double tmp = 0.0;
			int index_this = 0;
			for (int j = 0; j < nmopi[0]; ++j) {
				if (fabs(r->get(j)) > tmp) {
					index_this = j;  //find the largest C index, record row number (AO index!)
					tmp = r->get(0, j);
				}
			}
			if (index_this < natom_sys) {  //Change this if symmetry beyond c1 is allowed in the future
											 //r belongs to system!
				index_trace.push_back(i); //record index of MO column
				outfile->Printf("\n find %d", i);
				++count_orbs;
			}
			//r belongs to environment, do nothing
		}
	}

	if (methods == 2) { //Use IAO labels directly!
		int thres = natom_sys + 1;
		for (int i = 0; i < nmopi[0]; ++i) { // Change this if symmetry beyond c1 is allowed in the future
			int tmp = Loc["I"]->get(0, i, i);
			if (tmp < thres) {
				index_trace.push_back(i);
				++count_orbs;
			}
		}
	}

	//Use index_trace to generate C_sys
	Dimension nmo_sys_pi = nmopi;
	Dimension nbasis_sys_pi = nmopi;
	for (int h = 0; h < nirrep; ++h) {
		nmo_sys_pi[h] = count_orbs; //Change this if symmetry beyond c1 is allowed in the future
	}
	for (int h = 0; h < nirrep; ++h) {
		nbasis_sys_pi[h] = count_basis; //Change this if symmetry beyond c1 is allowed in the future
	}
	//int nempty = nbf - nmo_sys_pi[0];
	SharedMatrix C_sys(new Matrix("System Coeffs", nirrep, nmopi, nmo_sys_pi));
	//add loops here if symmetry beyond c1 is allowed in the future
	for (int k = 0; k < count_orbs; ++k) {
		SharedVector r = Loc["IAO"]->get_column(0, index_trace[k]);
		outfile->Printf("\n ** the %dth IAO coeffs on \"system\" atom found, index is %d ** ", k, index_trace[k]);
		r->print();

		/*
		//project or shrink r to small basis
		SharedVector r_short(new Vector("MO coeffs in small basis", nbasis_sys_pi));
		for (int l = 0; l < count_basis; ++l) {
			r_short->set(0, l, r->get(0, l));
		}
		*/
		C_sys->set_column(0, k, r);
	}
	outfile->Printf("\n ** IAO coeffs on \"system\" atoms (AO basis) ** \n");
	C_sys->print();

	//SharedWavefunction wfn_A(new Wavefunction(mol_sys, wfn->basisset()));
	//SharedMatrix C_new = wfn_A->Ca();
	//C_new->print();

	return C_sys; //change return value to maps later
}

double Embedding::do_env(SharedWavefunction wfn, SharedMatrix hcore, int methods, Options& options) {
	//Add options for environment methods here
	//Currently HF, add functionals in the future
	int nirrep = wfn->nirrep();
	Dimension nmopi = wfn->nmopi();
	Dimension noccpi = wfn->doccpi();

	if (methods == 0) { //method = HF, add to options later
		double E_conv = 1e-10; //convergence control for energy
		int Maxcyc = 100;

		//run HF calculation for environment
		double E = do_HF(wfn, hcore, nirrep, nmopi, noccpi, Maxcyc, E_conv, options);
		
		return E;
	}

	//add DFT calculation for environment here
	return 0.0;
}

double Embedding::do_sys(SharedWavefunction wfn, SharedMatrix h_ainb, int methods, Options& options) {
	//Add options for system methods here
	//Currently HF, add WFs in the future
	int nirrep = wfn->nirrep();
	Dimension nmopi = wfn->nmopi();
	Dimension noccpi = wfn->doccpi();

	if (methods == 0) { //method = HF, add to options later
		double E_conv = 1e-10; //convergence control for energy
		int Maxcyc = 100;

		//run HF calculation for environment
		double E = do_HF(wfn, h_ainb, nirrep, nmopi, noccpi, Maxcyc, E_conv, options);

		return E;
	}

	//add other expensive wavefunction methods here
	return 0.0;
}

std::map<std::string, SharedMatrix> Embedding::localize(SharedWavefunction wfn, int methods, Options& options) {
	//Add options for localization here
	int nirrep = wfn->nirrep();
	Dimension nmopi = wfn->nmopi();
	Dimension noccpi = wfn->doccpi();

	if (methods == 0) {
		//IAO localization
		outfile->Printf("\n Coeffs before localization \n");
		wfn->Ca()->print();
		std::shared_ptr<IAOBuilder> iaobd;
		iaobd = IAOBuilder::build(wfn->basisset(), wfn->get_basisset("MINAO_BASIS"), wfn->Ca(), options);
		std::map<std::string, SharedMatrix> ret = iaobd->build_iaos();
		outfile->Printf("\n ------ IAO coeffs! ------ \n");
		ret["A"]->print();
		//outfile->Printf("iao build U");
		//ret["U"]->print();

		SharedMatrix Cocc = build_Cocc(wfn->Ca(), nirrep, nmopi, noccpi);
		SharedMatrix Focc = build_Focc(wfn->Fa(), nirrep, nmopi, noccpi);
		std::vector<int> ranges = {};
		std::map<std::string, SharedMatrix> ret_loc = iaobd->localize(Cocc, Focc, ranges);
		//outfile->Printf("iao build C localized\n");
		//ret_loc["L_local"]->print();
		//outfile->Printf("iao build localized charge matrix \n");
		//ret_loc["Q"]->print();

		std::vector<std::string> iaolabel = iaobd->print_IAO(ret["A"], ret_loc["U"]->colspi()[0], wfn->basisset()->nbf(), wfn);

		int sizelab = iaolabel.size();
		std::vector<int> index_label = {};
		for (int i = 0; i < sizelab; ++i) {
			std::string tmps = iaolabel[i];
			char tmp = tmps.at(0);
			int tmpint = tmp;
			tmpint -= 48;
			outfile->Printf("\n IAO orbital: %s, Atom number %d \n", tmps.c_str(), tmpint);
			index_label.push_back(tmpint);
		}

		SharedMatrix Ind(new Matrix("Index Matrix", nirrep, nmopi, nmopi));
		for (int i = 0; i < nmopi[0]; ++i) {
			Ind->set(0, i, i, index_label[i]);
		}

		ret_loc["I"] = Ind;
		ret_loc["IAO"] = ret["A"];
		ret_loc["Trans"] = ret["U"];

		return ret_loc;
	}

	//Other localization methods go here
}

double Embedding::compute_energy() {
		// Initialize calculation ~
		// Require a molecule in wfn object with subset A and B!
		std::shared_ptr<Molecule> mol = ref_wfn_->molecule();
		int nfrag = mol->nfragments();

		if (nfrag == 1) {
			outfile->Printf("Warning! A input molecule with fragments (--- in atom list) is required for embedding!");
		}

		outfile->Printf("\n The input molecule have %d fragments, assigning the first fragment as system! \n", nfrag);

		std::vector<int> none_list = {};
		std::vector<int> sys_list = { 0 };
		std::vector<int> env_list = { 1 }; //change to 1-nfrag in the future!

		std::shared_ptr<Molecule> mol_sys = mol->extract_subsets(sys_list, none_list);
		std::shared_ptr<Molecule> mol_env = mol->extract_subsets(env_list, none_list);
		outfile->Printf("\n System Fragment \n");
		mol_sys->print();
		outfile->Printf("\n Environment Fragment(s) \n");
		mol_env->print();

		int natom_sys = mol_sys->natom();

		//1. Compute (environment method) energy for the whole system, currently HF
		SharedMatrix h_tot = ref_wfn_->H();
		double E_tot = do_env(ref_wfn_, h_tot, 0, options_); //put method = 2 to skip this step now!

		//2. Localize orbitals to IAO, Compute (environment method) energy for the whole system in IAO
		std::map<std::string, SharedMatrix> Loc = localize(ref_wfn_, 0, options_); //put method = 1 to skip this step!

		//3. Evaluate C, decide which orbital belong to which subset, A or B, based on input molecule subsets
		//SharedMatrix C_B(new Matrix("C_local", nirrep_, nmopi_, nmopi_));
		SharedMatrix C_A = Separate_orbital_subset(ref_wfn_, natom_sys, Loc, 2);

		C_A->print();
		//4. create wfn_A with basis and C_A
		//SharedWavefunction wfn_env(new Wavefunction(molecule_, basisset_));
		//wfn_env->deep_copy(ref_wfn_); //Save a copy of the original wfn!!
		SharedMatrix C_origin((new Matrix("Saved Coeffs", nirrep_, nmopi_, nmopi_)));
		C_origin->copy(ref_wfn_->Ca());
		SharedMatrix S_origin((new Matrix("Saved Overlaps", nirrep_, nmopi_, nmopi_)));
		S_origin->copy(ref_wfn_->S());
		SharedMatrix H_origin((new Matrix("Saved Hcore", nirrep_, nmopi_, nmopi_)));
		H_origin->copy(ref_wfn_->H());
		SharedMatrix Fa_origin((new Matrix("Saved Fock", nirrep_, nmopi_, nmopi_)));
		Fa_origin->copy(ref_wfn_->Fa());
	
		//From here, change the ref_wfn_ to wfn of system!!!
		//molecule_->deactivate_all_fragments();
		//molecule_->set_active_fragment(0);

		outfile->Printf("\n ****** Setting up system wavefunction ****** \n");

		//set Ca
		Ca_->copy(C_A);

		//rotate S
		SharedMatrix S_sys = Matrix::triplet(Loc["Trans"], S_origin, Loc["Trans"], true, false, false);
		S_->copy(S_sys);

		//rotate h
		SharedMatrix h_sys = Matrix::triplet(Loc["Trans"], H_origin, Loc["Trans"], true, false, false);
		H_->copy(h_sys); //wfn now have a iao based h_sys without embedding

		//rotate F
		SharedMatrix F_sys = Matrix::triplet(Loc["Trans"], Fa_origin, Loc["Trans"], true, false, false);
		Fa_->copy(F_sys);

		//set molecule
		//molecule_ = mol_sys;
		//(*molecule_) = *mol_sys;
		mol_sys->print();
		ref_wfn_->molecule() = mol_sys;
		ref_wfn_->molecule()->print();

		//Now ref_wfn_ has been wfn of system (with full basis)!
		outfile->Printf("\n ****** System wavefunction Set! ****** \n");
		ref_wfn_->molecule()->print();
		ref_wfn_->S()->print();
		ref_wfn_->Ca()->print();

		//5. Evaluate G(A + B) and G(A), evaluate and project h A-in-B (function)
		SharedMatrix Ga(new Matrix("G_sys", nirrep_, nmopi_, nmopi_));
		build_G(ref_wfn_, Ca_, Ga, options_, 0); //build G(A)
		outfile->Printf("\n ------ Building system G(A) ------ \n");
		Ca_->print();
		Ga->print();

		SharedMatrix Gab(new Matrix("G_all", nirrep_, nmopi_, nmopi_));
		build_G(ref_wfn_, C_origin, Gab, options_, 0); //build G(A+B)
		outfile->Printf("\n ------ Building system and environment G(A+B) ------ \n");
		C_origin->print();
		Gab->print();

		SharedMatrix C_iao_all = Matrix::doublet(C_origin, Loc["Trans"], false, false);
		SharedMatrix Gab_iao(new Matrix("G_all_rotate", nirrep_, nmopi_, nmopi_));
		outfile->Printf("\n ------ Checkpoint: iao G(A+B), should equal ------ \n");
		build_G(ref_wfn_, C_iao_all, Gab_iao, options_, 0); //build G(A+B) in iao, test whether they are the same!
		Gab_iao->print();

		outfile->Printf("\n ------ Building H A-in-B ------ \n");
		h_sys->add(Gab);
		h_sys->subtract(Ga);  //build h A-in-B !!
		h_sys->print();

		//6. Compute (cheap environment method) energy for system A, with h A-in-B
		outfile->Printf("\n ------ calculating E_sys_cheap ------ \n");
		double E_sys_cheap = do_env(ref_wfn_, h_sys, 0, options_);

		//7. Compute （expensive system method) energy for system A, with h A-in-B
		outfile->Printf("\n ------ calculating E_sys_exp ------ \n");
		double E_sys_exp = do_sys(ref_wfn_, h_sys, 0, options_);

		// Return Energy
		double E_embedding = E_tot - E_sys_cheap + E_sys_exp;
		outfile->Printf("\n ------ (insert method string here) Embedding Energy: E = %8.8f ------ \n", E_embedding);
        return E_embedding;
}

Embedding::~Embedding() {
}
}
}

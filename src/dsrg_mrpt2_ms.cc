#include "dsrg_mrpt2.h"
#include "ci_rdms.h"
#include "fci_solver.h"

namespace psi{ namespace forte{

double DSRG_MRPT2::compute_energy_multi_state(){
    // get character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for(int h = 0; h < this->nirrep(); ++h){
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // multi-state calculation
    std::vector<std::vector<double>> Edsrg_ms;

    if(options_.get_str("DSRG_SA_HEFF") == "XMS"){
        Edsrg_ms = compute_energy_xms();
    }else{
        Edsrg_ms = compute_energy_sa();
    }

    // energy summuary
    print_h2("Multi-State DSRG-MRPT2 Energy Summary");

    outfile->Printf("\n    Multi.  Irrep.  No.    DSRG-MRPT2 Energy");
    std::string dash(41, '-');
    outfile->Printf("\n    %s", dash.c_str());

    int nentry = eigens_.size();
    for(int n = 0; n < nentry; ++n){
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = options_["AVG_STATE"][n][2].to_integer();

        for(int i = 0; i < nstates; ++i){
            outfile->Printf("\n     %3d     %3s    %2d   %20.12f",
                            multi, irrep_symbol[irrep].c_str(), i, Edsrg_ms[n][i]);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }

    Process::environment.globals["CURRENT ENERGY"] = Edsrg_ms[0][0];
    return Edsrg_ms[0][0];
}

std::vector<std::vector<double>> DSRG_MRPT2::compute_energy_sa(){
    // compute DSRG-MRPT2 energy
    compute_energy();

    // transfer integrals
    transfer_integrals();

    // prepare FCI integrals
    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, aactv_mos_, acore_mos_);
    fci_ints->set_active_integrals(Hbar2_.block("aaaa"), Hbar2_.block("aAaA"), Hbar2_.block("AAAA"));
    fci_ints->compute_restricted_one_body_operator();

    // get character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for(int h = 0; h < this->nirrep(); ++h){
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // multiplicity table
    std::vector<std::string> multi_label{"Singlet","Doublet","Triplet","Quartet","Quintet","Sextet","Septet","Octet",
                                  "Nonet","Decaet","11-et","12-et","13-et","14-et","15-et","16-et","17-et","18-et",
                                  "19-et","20-et","21-et","22-et","23-et","24-et"};

    // size of 1rdm and 2rdm
    size_t na = mo_space_info_->size("ACTIVE");
    size_t nele1 = na * na;
    size_t nele2 = nele1 * nele1;

    // get effective one-electron integral (DSRG transformed)
    BlockedTensor oei = BTF_->build(tensor_type_,"temp1",spin_cases({"aa"}));
    oei.block("aa").data() = fci_ints->oei_a_vector();
    oei.block("AA").data() = fci_ints->oei_b_vector();

    // get nuclear repulsion energy
    std::shared_ptr<Molecule> molecule = Process::environment.molecule();
    double Enuc = molecule->nuclear_repulsion_energy();

    // loop over entries of AVG_STATE
    print_h2("Diagonalize Effective Hamiltonian");
    outfile->Printf("\n");
    int nentry = eigens_.size();
    std::vector<std::vector<double>> Edsrg_sa (nentry, std::vector<double> ());

    for(int n = 0; n < nentry; ++n){
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = options_["AVG_STATE"][n][2].to_integer();

        // diagonalize which the second-order effective Hamiltonian
        // FULL: CASCI using determinants
        // AVG_STATES: H_AB = <A|H|B> where A and B are SA-CAS states
        if(options_.get_str("DSRG_SA_HEFF") == "FULL") {

            outfile->Printf("    Use string FCI code.");

            // prepare FCISolver
            int charge = Process::environment.molecule()->molecular_charge();
            if(options_["CHARGE"].has_changed()){
                charge = options_.get_int("CHARGE");
            }
            auto nelec = 0;
            int natom = Process::environment.molecule()->natom();
            for(int i = 0; i < natom; ++i){
                nelec += Process::environment.molecule()->fZ(i);
            }
            nelec -= charge;
            int ms = (multi + 1) % 2;
            auto nelec_actv = nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * acore_mos_.size();
            auto na = (nelec_actv + ms) / 2;
            auto nb =  nelec_actv - na;

            Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
            int ntrial_per_root = options_.get_int("NTRIAL_PER_ROOT");

            FCISolver fcisolver(active_dim,acore_mos_,aactv_mos_,na,nb,multi,irrep,
                                ints_,mo_space_info_,ntrial_per_root,print_,options_);
            fcisolver.set_max_rdm_level(1);
            fcisolver.set_nroot(nstates);
            fcisolver.set_root(nstates - 1);
            fcisolver.set_fci_iterations(options_.get_int("FCI_ITERATIONS"));
            fcisolver.set_collapse_per_root(options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
            fcisolver.set_subspace_per_root(options_.get_int("DAVIDSON_SUBSPACE_PER_ROOT"));

            // compute energy and fill in results
            fcisolver.compute_energy();
            SharedVector Ems = fcisolver.eigen_vals();
            for(int i = 0; i < nstates; ++i){
                Edsrg_sa[n].push_back(Ems->get(i) + Enuc);
            }

        } else {

            int dim = (eigens_[n][0].first)->dim();
            SharedMatrix evecs (new Matrix("evecs",dim,dim));
            for(int i = 0; i < eigens_[n].size(); ++i){
                evecs->set_column(0,i,(eigens_[n][i]).first);
            }

            SharedMatrix Heff (new Matrix("Heff " + multi_label[multi - 1]
                               + " " + irrep_symbol[irrep], nstates, nstates));
            for(int A = 0; A < nstates; ++A){
                for(int B = A; B < nstates; ++B){

                    // compute rdms
                    CI_RDMS ci_rdms (options_,fci_ints,p_space_,evecs,A,B);
                    ci_rdms.set_symmetry(irrep);

                    std::vector<double> opdm_a (nele1, 0.0);
                    std::vector<double> opdm_b (nele1, 0.0);
                    ci_rdms.compute_1rdm(opdm_a,opdm_b);

                    std::vector<double> tpdm_aa (nele2, 0.0);
                    std::vector<double> tpdm_ab (nele2, 0.0);
                    std::vector<double> tpdm_bb (nele2, 0.0);
                    ci_rdms.compute_2rdm(tpdm_aa,tpdm_ab,tpdm_bb);

                    // put rdms in tensor format
                    BlockedTensor D1 = BTF_->build(tensor_type_,"D1",spin_cases({"aa"}),true);
                    D1.block("aa").data() = opdm_a;
                    D1.block("AA").data() = opdm_b;

                    BlockedTensor D2 = BTF_->build(tensor_type_,"D2",spin_cases({"aaaa"}),true);
                    D2.block("aaaa").data() = tpdm_aa;
                    D2.block("aAaA").data() = tpdm_ab;
                    D2.block("AAAA").data() = tpdm_bb;

                    double H_AB = 0.0;
                    H_AB += oei["uv"] * D1["uv"];
                    H_AB += oei["UV"] * D1["UV"];
                    H_AB += 0.25 * Hbar2_["uvxy"] * D2["xyuv"];
                    H_AB += 0.25 * Hbar2_["UVXY"] * D2["XYUV"];
                    H_AB += Hbar2_["uVxY"] * D2["xYuV"];

                    if(A == B){
                        H_AB += ints_->frozen_core_energy() + fci_ints->scalar_energy() + Enuc;
                        Heff->set(A,B,H_AB);
                    } else {
                        Heff->set(A,B,H_AB);
                        Heff->set(B,A,H_AB);
                    }

                }
            } // end forming effective Hamiltonian

            SharedMatrix U (new Matrix("U of Heff", nstates, nstates));
            SharedVector Ems (new Vector("MS Energies", nstates));
            Heff->diagonalize(U, Ems);

            Heff->print();
            //        U->eivprint(Ems);

            // fill in Edsrg_sa
            for(int i = 0; i < nstates; ++i){
                Edsrg_sa[n].push_back(Ems->get(i));
            }
        } // end if DSRG_AVG_DIAG

    } // end looping averaged states

    return Edsrg_sa;
}

std::vector<std::vector<double>> DSRG_MRPT2::compute_energy_xms(){
    // get character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for(int h = 0; h < this->nirrep(); ++h){
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // multiplicity table
    std::vector<std::string> multi_label{"Singlet","Doublet","Triplet","Quartet","Quintet","Sextet","Septet","Octet",
                                  "Nonet","Decaet","11-et","12-et","13-et","14-et","15-et","16-et","17-et","18-et",
                                  "19-et","20-et","21-et","22-et","23-et","24-et"};

    // obtain zeroth-order states
    int nentry = eigens_.size();
    std::vector<std::vector<double>> Edsrg_ms (nentry, std::vector<double> ());

    // prepare FCI integrals
    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, aactv_mos_, acore_mos_);
    ambit::Tensor actv_aa = ints_->aptei_aa_block(aactv_mos_, aactv_mos_, aactv_mos_, aactv_mos_);
    ambit::Tensor actv_ab = ints_->aptei_ab_block(aactv_mos_, aactv_mos_, aactv_mos_, aactv_mos_);
    ambit::Tensor actv_bb = ints_->aptei_bb_block(aactv_mos_, aactv_mos_, aactv_mos_, aactv_mos_);
    fci_ints->set_active_integrals(actv_aa, actv_ab, actv_bb);
    fci_ints->compute_restricted_one_body_operator();

    for(int n = 0; n < nentry; ++n){
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = eigens_[n].size();

        // fill in ci vectors
        int dim = (eigens_[n][0].first)->dim();
        SharedMatrix civecs (new Matrix("ci vecs",dim,nstates));
        for(int i = 0; i < nstates; ++i){
            civecs->set_column(0,i,(eigens_[n][i]).first);
        }

        // build Fock matrix
        SharedMatrix Fock (new Matrix("Fock", nstates, nstates));

        for(int M = 0; M < nstates; ++M){
            for(int N = M; N < nstates; ++N){

                // compute transition density
                CI_RDMS ci_rdms (options_,fci_ints,p_space_,civecs,M,N);
                ci_rdms.set_symmetry(irrep);

                std::vector<double> opdm_a,opdm_b;
                ci_rdms.compute_1rdm(opdm_a,opdm_b);

                // put rdms in tensor format
                BlockedTensor D1 = BTF_->build(tensor_type_,"D1",spin_cases({"aa"}),true);
                D1.block("aa").data() = opdm_a;
                D1.block("AA").data() = opdm_b;

                // compute Fock elements
                double F_MN = 0.0;
                F_MN += D1["uv"] * F_["vu"];
                F_MN += D1["UV"] * F_["VU"];
                Fock->set(M,N,F_MN);
                if(M != N){
                    Fock->set(N,M,F_MN);
                }
            }
        }

        SharedMatrix Fevec (new Matrix("Fock Evec", nstates, nstates));
        SharedVector Feval (new Vector("Fock Eval", nstates));
        Fock->diagonalize(Fevec, Feval);
        Fock->print();
        Fevec->eivprint(Feval);

        // Rotate ci vecs
        SharedMatrix rcivecs (new Matrix("rotated ci vecs",dim,nstates));
        rcivecs->gemm(false,false,1.0,civecs,Fevec,1.0);


        // prepare effective Hamiltonian
        SharedMatrix Heff (new Matrix("Heff " + multi_label[multi - 1] + " "
                           + irrep_symbol[irrep], nstates, nstates));

        // loop over nstates (with the same irrep and multi)
        for(int N = 0; N < nstates; ++N){

            // compute rdm and fill into cumulants
            compute_cumulants(fci_ints,rcivecs,N,N,irrep);

            // rebuild Fock matrix
            build_fock();

            // compute DSRG-MRPT2 energy
            // need to turn off semicanonical
            // TODO: solve amplitudes iteratively due to Factv off-diagonal elements
//            set_ignore_semicanonical(true);
            Eref_ = eigens_[n][N].second;
            double Edsrg = compute_energy();

            // TODO: save a copy of amplitudes

            // fill in diagonal Heff
            Heff->set(N,N,Edsrg);

            // rebuild integrals (because V is now renormalized)
            build_ints();
        }

        // get one-electron integrals
        BlockedTensor oei = BTF_->build(tensor_type_,"oei",spin_cases({"aa"}));
        oei.block("aa").data() = fci_ints->oei_a_vector();
        oei.block("AA").data() = fci_ints->oei_b_vector();

        // compute off-diagonal Heff
        for(int M = 0; M < nstates; ++M){
            for(int N = M + 1; N < nstates; ++N){

                // compute transition density
                CI_RDMS ci_rdms (options_,fci_ints,p_space_,rcivecs,M,N);
                ci_rdms.set_symmetry(irrep);

                std::vector<double> opdm_a,opdm_b;
                std::vector<double> tpdm_aa,tpdm_ab,tpdm_bb;

                ci_rdms.compute_1rdm(opdm_a,opdm_b);
                ci_rdms.compute_2rdm(tpdm_aa,tpdm_ab,tpdm_bb);

                // put rdms in tensor format
                BlockedTensor D1 = BTF_->build(tensor_type_,"D1",spin_cases({"aa"}),true);
                D1.block("aa").data() = opdm_a;
                D1.block("AA").data() = opdm_b;

                BlockedTensor D2 = BTF_->build(tensor_type_,"D2",spin_cases({"aaaa"}),true);
                D2.block("aaaa").data() = tpdm_aa;
                D2.block("aAaA").data() = tpdm_ab;
                D2.block("AAAA").data() = tpdm_bb;

                // first-order off-diagonal elements
                double H_MN = 0.0;
                H_MN += oei["uv"] * D1["uv"];
                H_MN += oei["UV"] * D1["UV"];
                H_MN += 0.25 * actv_aa("uvxy") * D2.block("aaaa")("xyuv");
                H_MN += 0.25 * actv_bb("UVXY") * D2.block("AAAA")("XYUV");
                H_MN += actv_ab("uVxY") * D2.block("aAaA")("xYuV");

                // TODO: compute second-order off-diagonal elements

                // fill in to Heff
                Heff->set(M,N,H_MN);
                Heff->set(N,M,H_MN);
            }
        }

        SharedMatrix U (new Matrix("U of Heff", nstates, nstates));
        SharedVector Ems (new Vector("MS Energies", nstates));
        Heff->diagonalize(U, Ems);

        Heff->print();
//        U->eivprint(Ems);

        // fill in Edsrg_ms
        for(int i = 0; i < nstates; ++i){
            Edsrg_ms[n].push_back(Ems->get(i));
        }
    }

    return Edsrg_ms;
}

void DSRG_MRPT2::compute_cumulants(std::shared_ptr<FCIIntegrals> fci_ints,
                                   SharedMatrix evecs, const int& root1, const int& root2,
                                   const int& irrep)
{
    CI_RDMS ci_rdms (options_,fci_ints,p_space_,evecs,root1,root2);
    ci_rdms.set_symmetry(irrep);

    std::vector<double> opdm_a, opdm_b;
    std::vector<double> tpdm_aa,tpdm_ab,tpdm_bb;
    std::vector<double> tpdm_aaa,tpdm_aab,tpdm_abb,tpdm_bbb;

    ci_rdms.compute_1rdm(opdm_a,opdm_b);
    ci_rdms.compute_2rdm(tpdm_aa,tpdm_ab,tpdm_bb);
    ci_rdms.compute_3rdm(tpdm_aaa,tpdm_aab,tpdm_abb,tpdm_bbb);

    // 1 cumulant
    ambit::Tensor L1a = Gamma1_.block("aa");
    ambit::Tensor L1b = Gamma1_.block("AA");

    L1a.data() = opdm_a;
    L1b.data() = opdm_b;

    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    Eta1_.block("aa")("pq") -= Gamma1_.block("aa")("pq");
    Eta1_.block("AA")("pq") -= Gamma1_.block("AA")("pq");

    // 2 cumulant
    ambit::Tensor L2aa = Lambda2_.block("aaaa");
    ambit::Tensor L2ab = Lambda2_.block("aAaA");
    ambit::Tensor L2bb = Lambda2_.block("AAAA");

    L2aa.data() = tpdm_aa;
    L2ab.data() = tpdm_ab;
    L2bb.data() = tpdm_bb;

    L2aa("pqrs") -= L1a("pr") * L1a("qs");
    L2aa("pqrs") += L1a("ps") * L1a("qr");

    L2bb("pqrs") -= L1b("pr") * L1b("qs");
    L2bb("pqrs") += L1b("ps") * L1b("qr");

    L2ab("pqrs") -= L1a("pr") * L1b("qs");

    // 3 cumulant
    ambit::Tensor L3aaa = Lambda3_.block("aaaaaa");
    ambit::Tensor L3aab = Lambda3_.block("aaAaaA");
    ambit::Tensor L3abb = Lambda3_.block("aAAaAA");
    ambit::Tensor L3bbb = Lambda3_.block("AAAAAA");

    L3aaa.data() = tpdm_aaa;
    L3aab.data() = tpdm_aab;
    L3abb.data() = tpdm_abb;
    L3bbb.data() = tpdm_bbb;

    // - step 1: aaa
    L3aaa("pqrstu") -= L1a("ps") * L2aa("qrtu");
    L3aaa("pqrstu") += L1a("pt") * L2aa("qrsu");
    L3aaa("pqrstu") += L1a("pu") * L2aa("qrts");

    L3aaa("pqrstu") -= L1a("qt") * L2aa("prsu");
    L3aaa("pqrstu") += L1a("qs") * L2aa("prtu");
    L3aaa("pqrstu") += L1a("qu") * L2aa("prst");

    L3aaa("pqrstu") -= L1a("ru") * L2aa("pqst");
    L3aaa("pqrstu") += L1a("rs") * L2aa("pqut");
    L3aaa("pqrstu") += L1a("rt") * L2aa("pqsu");

    L3aaa("pqrstu") -= L1a("ps") * L1a("qt") * L1a("ru");
    L3aaa("pqrstu") -= L1a("pt") * L1a("qu") * L1a("rs");
    L3aaa("pqrstu") -= L1a("pu") * L1a("qs") * L1a("rt");

    L3aaa("pqrstu") += L1a("ps") * L1a("qu") * L1a("rt");
    L3aaa("pqrstu") += L1a("pu") * L1a("qt") * L1a("rs");
    L3aaa("pqrstu") += L1a("pt") * L1a("qs") * L1a("ru");

    // - step 2: aab
    L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
    L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

    L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
    L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

    L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

    L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
    L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");

    // - step 3: abb
    L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

    L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
    L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

    L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
    L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

    L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
    L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");

    // - step 4: bbb
    L3bbb("pqrstu") -= L1b("ps") * L2bb("qrtu");
    L3bbb("pqrstu") += L1b("pt") * L2bb("qrsu");
    L3bbb("pqrstu") += L1b("pu") * L2bb("qrts");

    L3bbb("pqrstu") -= L1b("qt") * L2bb("prsu");
    L3bbb("pqrstu") += L1b("qs") * L2bb("prtu");
    L3bbb("pqrstu") += L1b("qu") * L2bb("prst");

    L3bbb("pqrstu") -= L1b("ru") * L2bb("pqst");
    L3bbb("pqrstu") += L1b("rs") * L2bb("pqut");
    L3bbb("pqrstu") += L1b("rt") * L2bb("pqsu");

    L3bbb("pqrstu") -= L1b("ps") * L1b("qt") * L1b("ru");
    L3bbb("pqrstu") -= L1b("pt") * L1b("qu") * L1b("rs");
    L3bbb("pqrstu") -= L1b("pu") * L1b("qs") * L1b("rt");

    L3bbb("pqrstu") += L1b("ps") * L1b("qu") * L1b("rt");
    L3bbb("pqrstu") += L1b("pu") * L1b("qt") * L1b("rs");
    L3bbb("pqrstu") += L1b("pt") * L1b("qs") * L1b("ru");
}

}}

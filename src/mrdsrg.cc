#include <algorithm>
#include <vector>
#include <map>
#include <cmath>

#include <boost/format.hpp>

#include <libmints/molecule.h>

#include "helpers.h"
#include "mrdsrg.h"
#include "fci_solver.h"
#include "mp2_nos.h"

namespace psi{ namespace forte{

MRDSRG::MRDSRG(Reference reference, boost::shared_ptr<Wavefunction> wfn, Options& options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options,_default_psio_lib_), wfn_(wfn), reference_(reference), ints_(ints),
      mo_space_info_(mo_space_info), BTF_(new BlockedTensorFactory(options)), tensor_type_(CoreTensor)
{
    print_method_banner({"Multireference Driven Similarity Renormalization Group","Chenyang Li"});
    read_options();
    print_options();
    startup();
}

MRDSRG::~MRDSRG(){
    cleanup();
}

void MRDSRG::cleanup(){
    print_comm_time();
}

void MRDSRG::read_options(){

    print_ = options_.get_int("PRINT");

    source_ = options_.get_str("SOURCE");

    s_ = options_.get_double("DSRG_S");
    if(s_ < 0){
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        throw PSIEXCEPTION("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
    if(taylor_threshold_ <= 0){
        outfile->Printf("\n  Threshold for Taylor expansion must be an integer greater than 0!");
        throw PSIEXCEPTION("Threshold for Taylor expansion must be an integer greater than 0!");
    }
    taylor_order_ = int(0.5 * (15.0 / taylor_threshold_ + 1)) + 1;

    ntamp_ = options_.get_int("NTAMP");
    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");
}

void MRDSRG::startup()
{
    // frozen-core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // reference energy
    Eref_ = reference_.get_Eref();

    // orbital spaces
    BlockedTensor::reset_mo_spaces();
    acore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bcore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    bactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    acore_label_ = "c";
    aactv_label_ = "a";
    avirt_label_ = "v";
    bcore_label_ = "C";
    bactv_label_ = "A";
    bvirt_label_ = "V";
    BTF_->add_mo_space(acore_label_,"mn",acore_mos_,AlphaSpin);
    BTF_->add_mo_space(bcore_label_,"MN",bcore_mos_,BetaSpin);
    BTF_->add_mo_space(aactv_label_,"uvwxyz",aactv_mos_,AlphaSpin);
    BTF_->add_mo_space(bactv_label_,"UVWXYZ",bactv_mos_,BetaSpin);
    BTF_->add_mo_space(avirt_label_,"ef",avirt_mos_,AlphaSpin);
    BTF_->add_mo_space(bvirt_label_,"EF",bvirt_mos_,BetaSpin);

    // map space labels to mo spaces
    label_to_spacemo_[acore_label_[0]] = acore_mos_;
    label_to_spacemo_[bcore_label_[0]] = bcore_mos_;
    label_to_spacemo_[aactv_label_[0]] = aactv_mos_;
    label_to_spacemo_[bactv_label_[0]] = bactv_mos_;
    label_to_spacemo_[avirt_label_[0]] = avirt_mos_;
    label_to_spacemo_[bvirt_label_[0]] = bvirt_mos_;

    // define composite spaces
    BTF_->add_composite_mo_space("h","ijkl",{acore_label_,aactv_label_});
    BTF_->add_composite_mo_space("H","IJKL",{bcore_label_,bactv_label_});
    BTF_->add_composite_mo_space("p","abcd",{aactv_label_,avirt_label_});
    BTF_->add_composite_mo_space("P","ABCD",{bactv_label_,bvirt_label_});
    BTF_->add_composite_mo_space("g","pqrsto",{acore_label_,aactv_label_,avirt_label_});
    BTF_->add_composite_mo_space("G","PQRSTO",{bcore_label_,bactv_label_,bvirt_label_});

    // prepare integrals
    H_ = BTF_->build(tensor_type_,"H",spin_cases({"gg"}));
    V_ = BTF_->build(tensor_type_,"V",spin_cases({"gggg"}));
    build_ints();

    // prepare density matrix and cumulants
    // TODO: future code will store only active Gamma1 and Eta1
    Eta1_ = BTF_->build(tensor_type_,"Eta1",spin_cases({"pp"}));
    Gamma1_ = BTF_->build(tensor_type_,"Gamma1",spin_cases({"hh"}));
    Lambda2_ = BTF_->build(tensor_type_,"Lambda2",spin_cases({"aaaa"}));
    if(options_.get_str("THREEPDC") != "ZERO"){
        Lambda3_ = BTF_->build(tensor_type_,"Lambda3",spin_cases({"aaaaaa"}));
    }
    build_density();

    // build Fock matrix
    F_ = BTF_->build(tensor_type_,"Fock",spin_cases({"gg"}));
    build_fock(H_,V_);

    // auto adjusted s_
    s_ = make_s_smart();

    // test semi-canonical
    print_h2("Checking Orbitals");
    H0th_ = BTF_->build(tensor_type_,"Zeroth-order H",diag_labels());
    H0th_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if(i[0] == i[1]){
            if(spin[0] == AlphaSpin){
                value = Fa_[i[0]];
            }else{
                value = Fb_[i[0]];
            }
        }
    });
    semi_canonical_ = check_semicanonical();
    if(!semi_canonical_){
        outfile->Printf("\n    MR-DSRG will be computed in an arbitrary basis. Orbital invariant formulasm is employed.");
        outfile->Printf("\n    We recommend using semi-canonical for all denominator-based source operator.");
        if(options_.get_str("RELAX_REF") != "NONE"){
            outfile->Printf("\n\n    Currently, only RELAX_REF = NONE is available for orbital invariant formalism.");
            throw PSIEXCEPTION("Orbital invariant formalism is not implemented for the RELAX_REF option.");
        }else{
            U_ = ambit::BlockedTensor::build(tensor_type_,"U",spin_cases({"gg"}));
            std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
            Fa_ = eigens[0];
            Fb_ = eigens[1];
        }
    }else{
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
    }
}

void MRDSRG::build_ints(){
    // prepare one-electron integrals
    H_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin) value = ints_->oei_a(i[0],i[1]);
        else value = ints_->oei_b(i[0],i[1]);
    });

    // prepare two-electron integrals
    V_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
        if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin))  value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin))  value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    });
}

void MRDSRG::build_density(){
    // prepare density matrices
    (Gamma1_.block("cc")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Gamma1_.block("CC")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1_.block("vv")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1_.block("VV")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    Gamma1_.block("aa")("pq") = reference_.L1a()("pq");
    Gamma1_.block("AA")("pq") = reference_.L1b()("pq");
    Eta1_.block("aa")("pq") -= reference_.L1a()("pq");
    Eta1_.block("AA")("pq") -= reference_.L1b()("pq");

    // prepare two-body density cumulants
    ambit::Tensor Lambda2_aa = Lambda2_.block("aaaa");
    ambit::Tensor Lambda2_aA = Lambda2_.block("aAaA");
    ambit::Tensor Lambda2_AA = Lambda2_.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    // prepare three-body density cumulants
    if(options_.get_str("THREEPDC") != "ZERO"){
        ambit::Tensor Lambda3_aaa = Lambda3_.block("aaaaaa");
        ambit::Tensor Lambda3_aaA = Lambda3_.block("aaAaaA");
        ambit::Tensor Lambda3_aAA = Lambda3_.block("aAAaAA");
        ambit::Tensor Lambda3_AAA = Lambda3_.block("AAAAAA");
        Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
        Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
        Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
        Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");
    }

    // check cumulants
    print_cumulant_summary();
}

void MRDSRG::build_fock(BlockedTensor& H, BlockedTensor& V){
    // build Fock matrix
    F_["pq"]  = H["pq"];
    F_["pq"] += V["pjqi"] * Gamma1_["ij"];
    F_["pq"] += V["pJqI"] * Gamma1_["IJ"];
    // symmetrize beta spin
    outfile->Printf("\n  Warning: I am forcing Fb = Fa to avoid spin symmetry breaking.");
    outfile->Printf("\n  If this is not desired, go to mrdsrg.cc around line 220.");
    std::vector<std::string> bs {"cc","ca","cv","ac","aa","av","vc","va","vv"};
    for(std::string& b: bs){
        std::string b_beta (1, toupper(b[0]));
        b_beta += std::string (1, toupper(b[1]));
        F_.block(b_beta).data() = F_.block(b).data();
    }
//    F_["PQ"]  = H["PQ"];
//    F_["PQ"] += V["jPiQ"] * Gamma1_["ij"];
//    F_["PQ"] += V["PJQI"] * Gamma1_["IJ"];

    // obtain diagonal elements of Fock matrix
    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo_);
    Fb_ = std::vector<double>(ncmo_);
    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            Fb_[i[0]] = value;
        }
    });
}

void MRDSRG::print_options()
{
    // fill in information
    std::vector<std::pair<std::string,int>> calculation_info{
        {"ntamp", ntamp_},
        {"diis_min_vecs", options_.get_int("DIIS_MIN_VECS")},
        {"diis_max_vecs", options_.get_int("DIIS_MAX_VECS")}};

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"flow parameter",s_},
        {"taylor expansion threshold",pow(10.0,-double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"corr_level", options_.get_str("CORR_LEVEL")},
        {"int_type", options_.get_str("INT_TYPE")},
        {"source operator", source_},
        {"smart_dsrg_s", options_.get_str("SMART_DSRG_S")},
        {"reference relaxation", options_.get_str("RELAX_REF")},
        {"dsrg transformation type", options_.get_str("DSRG_TRANS_TYPE")},
        {"core virtual source type", options_.get_str("CCVV_SOURCE")}};

    // print some information
    print_h2("Calculation Information");
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-35s %15d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-35s %15.3e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-35s %15s",str_dim.first.c_str(),str_dim.second.c_str());
    }
    outfile->Flush();
}

double MRDSRG::renormalized_denominator(const double& D){
    double Z = std::sqrt(s_) * D;
    if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
        return Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
    }else{
        return (1.0 - std::exp(-s_ * std::pow(D, 2.0))) / D;
    }
}

double MRDSRG::renormalized_denominator_labs(const double& D){
    double Z = s_ * D;
    if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
        return Taylor_Exp_Linear(Z, taylor_order_ * 2) * s_;
    }else{
        return (1.0 - std::exp(-s_ * std::fabs(D))) / D;
    }
}

double MRDSRG::compute_energy(){
    // build initial amplitudes
    print_h2("Build Initial Amplitude from DSRG-MRPT2");
    T1_ = BTF_->build(tensor_type_,"T1 Amplitudes",spin_cases({"hp"}));
    T2_ = BTF_->build(tensor_type_,"T2 Amplitudes",spin_cases({"hhpp"}));
    guess_t(V_,T2_,F_,T1_);

    // check initial amplitudes
    analyze_amplitudes("First-Order",T1_,T2_);

    // get reference energy
    double Etotal = Eref_;

    // compute energy
    switch (corrlevelmap[options_.get_str("CORR_LEVEL")]){
    case CORR_LV::LDSRG2:{
        Etotal += compute_energy_ldsrg2();
        break;
    }
    case CORR_LV::CEPA0:{
        Etotal += compute_energy_cepa0();
        break;
    }
    case CORR_LV::LDSRG2_P3:{
        break;
    }
    case CORR_LV::QDSRG2:{
        break;
    }
    case CORR_LV::QDSRG2_P3:{
        break;
    }
    case CORR_LV::PT3:{
        Etotal += compute_energy_pt3();
        break;
    }
    default:{
        Etotal += compute_energy_pt2();
    }}

    Process::environment.globals["CURRENT ENERGY"] = Etotal;
    return Etotal;
}

double MRDSRG::compute_energy_relaxed(){
    // setup for FCISolver
    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
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
    int multi = Process::environment.molecule()->multiplicity();
    if(options_["MULTIPLICITY"].has_changed()){
        multi = options_.get_int("MULTIPLICITY");
    }
    int ms = multi - 1;
    if(options_["MS"].has_changed()){
        ms = options_.get_int("MS");
    }
    auto nelec_actv = nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * acore_mos_.size();
    auto na = (nelec_actv + ms) / 2;
    auto nb =  nelec_actv - na;

    // reference relaxation
    double Edsrg = 0.0, Erelax = 0.0;
    std::string relax_algorithm = options_.get_str("RELAX_REF");

    if(relax_algorithm == "ONCE"){
        // compute energy with fixed ref.
        Edsrg = compute_energy();

        // transfer integrals
        transfer_integrals();

        // diagonalize the Hamiltonian
        FCISolver fcisolver(active_dim,acore_mos_,aactv_mos_,na,nb,multi,options_.get_int("ROOT_SYM"),ints_, mo_space_info_,
                                             options_.get_int("NTRIAL_PER_ROOT"),print_, options_);
        Erelax = fcisolver.compute_energy();

        // printing
        print_h2("MRDSRG Energy Summary");
        outfile->Printf("\n    %-30s = %22.15f", "MRDSRG Total Energy (fixed)", Edsrg);
        outfile->Printf("\n    %-30s = %22.15f", "MRDSRG Total Energy (relaxed)", Erelax);
        outfile->Printf("\n");
    }
    else if(relax_algorithm == "ITERATE"){
        // iteration variables
        int cycle = 0, maxiter = options_.get_int("MAXITER_RELAX_REF");
        double e_conv = options_.get_double("E_CONVERGENCE");
        std::vector<double> Edsrg_vec, Erelax_vec;
        std::vector<double> Edelta_dsrg_vec, Edelta_relax_vec;
        bool converged = false, failed = false;

        // start iteration
        do{
            // compute dsrg energy
            double Etemp = Edsrg;
            Edsrg = compute_energy();
            Edsrg_vec.push_back(Edsrg);
            double Edelta_dsrg = Edsrg - Etemp;
            Edelta_dsrg_vec.push_back(Edelta_dsrg);

            // transfer integrals (O1, Hbar2)
            transfer_integrals();

            // diagonalize the Hamiltonian
            FCISolver fcisolver(active_dim,acore_mos_,aactv_mos_,na,nb,multi,options_.get_int("ROOT_SYM"),ints_,mo_space_info_,options_);
            Etemp = Erelax;
            fcisolver.set_max_rdm_level(3);
            fcisolver.set_nroot(options_.get_int("NROOT"));
            fcisolver.set_root(options_.get_int("ROOT"));
            Erelax = fcisolver.compute_energy();
            Erelax_vec.push_back(Erelax);
            double Edelta_relax = Erelax - Etemp;
            Edelta_relax_vec.push_back(Edelta_relax);

            // obtain new reference
            reference_ = fcisolver.reference();

            // refill densities
            build_density();

            // build the new Fock matrix
            build_fock(H_,V_);

            // diagonal blocks of Fock
            H0th_.zero();
            H0th_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
                if(i[0] == i[1]){
                    if(spin[0] == AlphaSpin){
                        value = Fa_[i[0]];
                    }else{
                        value = Fb_[i[0]];
                    }
                }
            });

            // semi-canonicalize orbitals
            print_h2("Semi-canonicalize Orbitals");
            bool semi = check_semicanonical();
            if (options_.get_bool("SEMI_CANONICAL") && !semi){
                // set up timer
                std::vector<double> timings {0.0};
                Timer timer;

                // diagonalize blocks of Fock matrix
                BlockedTensor U = ambit::BlockedTensor::build(tensor_type_,"U",spin_cases({"gg"}));
                diagonalize_Fock_diagblocks(U);
                outfile->Printf("\n    %-47s %8.3f", "Timing for block-diagonalizing Fock matrix:", timer.get() - timings.back());
                timings.push_back(timer.get());

                // transform O1 to new basis and copy to ints_
                BlockedTensor O = ambit::BlockedTensor::build(tensor_type_,"Temp One",spin_cases({"gg"}));
                O["rs"] = U["rp"] * O1_["pq"] * U["sq"];
                O["RS"] = U["RP"] * O1_["PQ"] * U["SQ"];
                O.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
                    if (spin[0] == AlphaSpin){
                        ints_->set_oei(i[0],i[1],value,true);
                    }else{
                        ints_->set_oei(i[0],i[1],value,false);
                    }
                });

                // transform bare one-body Hamiltonian
                O["rs"] = U["rp"] * H_["pq"] * U["sq"];
                O["RS"] = U["RP"] * H_["PQ"] * U["SQ"];
                H_["pq"] = O["pq"];
                H_["PQ"] = O["PQ"];
                outfile->Printf("\n    %-47s %8.3f", "Timing for transforming one-electron integral:", timer.get() - timings.back());
                timings.push_back(timer.get());

                // transform Hbar2 to new basis and copy to ints_
                O = ambit::BlockedTensor::build(tensor_type_,"Temp Two",spin_cases({"gggg"}));
                O["tors"] = U["tp"] * U["oq"] * Hbar2_["pqrs"];
                O["tOrS"] = U["tp"] * U["OQ"] * Hbar2_["pQrS"];
                O["TORS"] = U["TP"] * U["OQ"] * Hbar2_["PQRS"];
                Hbar2_["pqot"] = O["pqrs"] * U["ts"] * U["or"];
                Hbar2_["pQoT"] = O["pQrS"] * U["TS"] * U["or"];
                Hbar2_["PQOT"] = O["PQRS"] * U["TS"] * U["OR"];
                Hbar2_.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
                    if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
                        ints_->set_tei(i[0],i[1],i[2],i[3],value,true,true);
                    }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
                        ints_->set_tei(i[0],i[1],i[2],i[3],value,true,false);
                    }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
                        ints_->set_tei(i[0],i[1],i[2],i[3],value,false,false);
                    }
                });
                ints_->update_integrals(false);

                // transform bare two-body Hamiltonian
                O["tors"] = U["tp"] * U["oq"] * V_["pqrs"];
                O["tOrS"] = U["tp"] * U["OQ"] * V_["pQrS"];
                O["TORS"] = U["TP"] * U["OQ"] * V_["PQRS"];
                V_["pqot"] = O["pqrs"] * U["ts"] * U["or"];
                V_["pQoT"] = O["pQrS"] * U["TS"] * U["or"];
                V_["PQOT"] = O["PQRS"] * U["TS"] * U["OR"];
                outfile->Printf("\n    %-47s %8.3f", "Timing for transforming two-electron integral:", timer.get() - timings.back());
                timings.push_back(timer.get());

                // diagonalize the Hamiltonian
                FCISolver fcisolver(active_dim,acore_mos_,aactv_mos_,na,nb,multi,options_.get_int("ROOT_SYM"),ints_,mo_space_info_,options_);
                fcisolver.set_max_rdm_level(3);
                fcisolver.set_nroot(options_.get_int("NROOT"));
                fcisolver.set_root(options_.get_int("ROOT"));
                Erelax = fcisolver.compute_energy();
                if(fabs(Erelax - Erelax_vec.back()) > 100.0 * e_conv){
                    throw PSIEXCEPTION("Semi-canonicalization failed.");
                }

                // obtain new reference
                reference_ = fcisolver.reference();

                // refill densities
                build_density();

                // rebuild Fock matrix
                build_fock(H_,V_);
            }

            // test convergence
            if(fabs(Edelta_dsrg) < e_conv && fabs(Edelta_relax) < e_conv){
                converged = true;
            }
            if(cycle > maxiter){
                outfile->Printf("\n\n    The reference relaxation does not converge in %d iterations! Quitting.\n", maxiter);
                converged = true;
                failed = true;
                outfile->Flush();
            }
            ++cycle;
        } while(!converged);

        print_h2("MRDSRG Reference Relaxation Summary");
        std::string indent(4, ' ');
        std::string dash(71, '-');
        std::string title;
        title += indent + str(boost::format("%5c  %=31s  %=31s\n")
                              % ' ' % "Fixed Ref. (a.u.)" % "Relaxed Ref. (a.u.)");
        title += indent + std::string (7, ' ') + std::string (31, '-') + "  " + std::string (31, '-') + "\n";
        title += indent + str(boost::format("%5s  %=20s %=10s  %=20s %=10s\n")
                              % "Iter." % "Total Energy" % "Delta" % "Total Energy" % "Delta");
        title += indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for(int n = 0; n != cycle; ++n){
            outfile->Printf("\n    %5d  %20.12f %10.3e  %20.12f %10.3e", n,
                            Edsrg_vec[n], Edelta_dsrg_vec[n], Erelax_vec[n], Edelta_relax_vec[n]);
            outfile->Flush();
        }
        outfile->Printf("\n    %s", dash.c_str());
        outfile->Printf("\n    %-30s = %23.15f", "MRDSRG Total Energy", Edsrg);
        outfile->Printf("\n    %-30s = %23.15f", "MRDSRG Total Energy (relaxed)", Erelax);
        outfile->Printf("\n");

        if(failed){
            throw PSIEXCEPTION("Reference relaxation process does not converge.");
        }
    }

    Process::environment.globals["CURRENT ENERGY"] = Erelax;
    return Erelax;
}

void MRDSRG::transfer_integrals(){
    // printing
    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

    // compute scalar term
    Timer t_scalar;
    std::string str = "Computing the scalar term   ...";
    outfile->Printf("\n    %-35s", str.c_str());
    double scalar0 = Eref_ + Hbar0_ - molecule_->nuclear_repulsion_energy()
            - ints_->frozen_core_energy();

    // scalar from Hbar1
    double scalar1 = 0.0;
    Hbar1_.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar1 -= value;
    });
    Hbar1_.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar1 -= value;
    });
    scalar1 -= Hbar1_["vu"] * Gamma1_["uv"];
    scalar1 -= Hbar1_["VU"] * Gamma1_["UV"];

    // scalar from Hbar2
    double scalar2 = 0.0;
    scalar2 -= 0.25 * Hbar2_["xyuv"] * Lambda2_["uvxy"];
    scalar2 -= 0.25 * Hbar2_["XYUV"] * Lambda2_["UVXY"];
    scalar2 -= Hbar2_["xYuV"] * Lambda2_["uVxY"];
    Hbar2_.block("cccc").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += 0.5 * value;
    });
    Hbar2_.block("cCcC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += value;
    });
    Hbar2_.block("CCCC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += 0.5 * value;
    });

    O1_.zero();
    O1_["pq"] += Hbar2_["puqv"] * Gamma1_["vu"];
    O1_["pq"] += Hbar2_["pUqV"] * Gamma1_["VU"];
    O1_["PQ"] += Hbar2_["uPvQ"] * Gamma1_["vu"];
    O1_["PQ"] += Hbar2_["PUQV"] * Gamma1_["VU"];
    O1_.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar2 += value;
    });
    O1_.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar2 += value;
    });
    scalar2 += 0.5 * Gamma1_["uv"] * Hbar2_["vyux"] * Gamma1_["xy"];
    scalar2 += 0.5 * Gamma1_["UV"] * Hbar2_["VYUX"] * Gamma1_["XY"];
    scalar2 += Gamma1_["uv"] * Hbar2_["vYuX"] * Gamma1_["XY"];

    double scalar = scalar0 + scalar1 + scalar2;
    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

    // compute one-body term
    Timer t_one;
    str = "Computing the one-body term ...";
    outfile->Printf("\n    %-35s", str.c_str());
    O1_.scale(-1.0);
    O1_["pq"] += Hbar1_["pq"];
    O1_["PQ"] += Hbar1_["PQ"];
    BlockedTensor temp = BTF_->build(tensor_type_,"temp",spin_cases({"cc"}));
    temp.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>&,double& value){
        if (i[0] == i[1]) value = 1.0;
    });
    O1_["pq"] -= Hbar2_["pmqn"] * temp["nm"];
    O1_["pq"] -= Hbar2_["pMqN"] * temp["NM"];
    O1_["PQ"] -= Hbar2_["mPnQ"] * temp["nm"];
    O1_["PQ"] -= Hbar2_["PMQN"] * temp["NM"];
    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

    // update integrals
    Timer t_int;
    str = "Updating integrals          ...";
    outfile->Printf("\n    %-35s", str.c_str());
    ints_->set_scalar(scalar);
    O1_.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if (spin[0] == AlphaSpin){
            ints_->set_oei(i[0],i[1],value,true);
        }else{
            ints_->set_oei(i[0],i[1],value,false);
        }
    });
    Hbar2_.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,true);
        }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,false);
        }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,false,false);
        }
    });
    outfile->Printf("  Done. Timing %10.3f s", t_int.get());

    // print scalar
    double scalar_include_fc = scalar + ints_->frozen_core_energy();
    print_h2("Scalar of the DSRG Hamiltonian (WRT True Vacuum)");
    outfile->Printf("\n    %-30s = %22.15f", "Scalar0", scalar0);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar1", scalar1);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar2", scalar2);
    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/O Frozen-Core", scalar);
    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/  Frozen-Core", scalar_include_fc);

    // test if de-normal-ordering is correct
    print_h2("Test De-Normal-Ordered Hamiltonian");
    double Etest = scalar_include_fc + molecule_->nuclear_repulsion_energy();

    double Etest1 = 0.0;
    O1_.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) Etest1 += value;
    });
    O1_.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) Etest1 += value;
    });
    Etest1 += O1_["uv"] * Gamma1_["vu"];
    Etest1 += O1_["UV"] * Gamma1_["VU"];

    double Etest2 = 0.0;
    Hbar2_.block("cccc").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += 0.5 * value;
    });
    Hbar2_.block("cCcC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += value;
    });
    Hbar2_.block("CCCC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += 0.5 * value;
    });

    Etest2 += Hbar2_["munv"] * temp["nm"] * Gamma1_["vu"];
    Etest2 += Hbar2_["uMvN"] * temp["NM"] * Gamma1_["vu"];
    Etest2 += Hbar2_["mUnV"] * temp["nm"] * Gamma1_["VU"];
    Etest2 += Hbar2_["MUNV"] * temp["NM"] * Gamma1_["VU"];

    Etest2 += 0.5 * Gamma1_["vu"] * Hbar2_["uxvy"] * Gamma1_["yx"];
    Etest2 += 0.5 * Gamma1_["VU"] * Hbar2_["UXVY"] * Gamma1_["YX"];
    Etest2 += Gamma1_["vu"] * Hbar2_["uXvY"] * Gamma1_["YX"];

    Etest2 += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
    Etest2 += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
    Etest2 += Hbar2_["uVxY"] * Lambda2_["xYuV"];

    Etest += Etest1 + Etest2;
    outfile->Printf("\n    %-30s = %22.15f", "One-Body Energy (after)", Etest1);
    outfile->Printf("\n    %-30s = %22.15f", "Two-Body Energy (after)", Etest2);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (after)", Etest);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

    if(fabs(Etest - Eref_ - Hbar0_) > 100.0 * options_.get_double("E_CONVERGENCE")){
        throw PSIEXCEPTION("De-normal-odering failed.");
    }else{
        ints_->update_integrals(false);
    }
}

void MRDSRG::reset_ints(BlockedTensor& H, BlockedTensor& V){
    ints_->set_scalar(0.0);
    H.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if (spin[0] == AlphaSpin){
            ints_->set_oei(i[0],i[1],value,true);
        }else{
            ints_->set_oei(i[0],i[1],value,false);
        }
    });
    V.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,true);
        }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,false);
        }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,false,false);
        }
    });
    ints_->update_integrals(false);
}


std::vector<std::vector<double>> MRDSRG::diagonalize_Fock_diagblocks(BlockedTensor& U){
    // diagonal blocks identifiers (C-A-V ordering)
    std::vector<std::string> blocks = diag_labels();

    // map MO space label to its Dimension
    std::map<std::string, Dimension> MOlabel_to_dimension;
    MOlabel_to_dimension[acore_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_dimension[aactv_label_] = mo_space_info_->get_dimension("ACTIVE");
    MOlabel_to_dimension[avirt_label_] = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    // eigen values to be returned
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Dimension corr = mo_space_info_->get_dimension("CORRELATED");
    std::vector<double> eigenvalues_a(ncmo, 0.0);
    std::vector<double> eigenvalues_b(ncmo, 0.0);

    // map MO space label to its offset Dimension
    std::map<std::string, Dimension> MOlabel_to_offset_dimension;
    int nirrep = corr.n();
    MOlabel_to_offset_dimension[acore_label_] = Dimension(std::vector<int> (nirrep, 0));
    MOlabel_to_offset_dimension[aactv_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_offset_dimension[avirt_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC") + mo_space_info_->get_dimension("ACTIVE");

    // figure out index
    auto fill_eigen = [&](std::string block_label, int irrep, std::vector<double> values){
        int h = irrep;
        size_t idx_begin = 0;
        while((--h) >= 0) idx_begin += corr[h];

        std::string label (1, tolower(block_label[0]));
        idx_begin += MOlabel_to_offset_dimension[label][irrep];

        bool spin_alpha = islower(block_label[0]);
        size_t nvalues = values.size();
        if(spin_alpha){
            for(size_t i = 0; i < nvalues; ++i){
                eigenvalues_a[i + idx_begin] = values[i];
            }
        }else{
            for(size_t i = 0; i < nvalues; ++i){
                eigenvalues_b[i + idx_begin] = values[i];
            }
        }
    };

    // diagonalize diagonal blocks
    for(const auto& block: blocks){
        size_t dim = F_.block(block).dim(0);
        if(dim == 0){
            continue;
        }else{
            std::string label (1, tolower(block[0]));
            Dimension space = MOlabel_to_dimension[label];
            int nirrep = space.n();

            // separate Fock with irrep
            for(int h = 0; h < nirrep; ++h){
                size_t h_dim = space[h];
                ambit::Tensor U_h;
                if(h_dim == 0){
                    continue;
                }else if(h_dim == 1){
                    U_h = ambit::Tensor::build(tensor_type_,"U_h",std::vector<size_t> (2, h_dim));
                    U_h.data()[0] = 1.0;
                    ambit::Tensor F_block = F_.block(block);
                    ambit::Tensor T_h = separate_tensor(F_block,space,h);
                    fill_eigen(block,h,T_h.data());
                }else{
                    ambit::Tensor F_block = F_.block(block);
                    ambit::Tensor T_h = separate_tensor(F_block,space,h);
                    auto Feigen = T_h.syev(AscendingEigenvalue);
                    U_h = ambit::Tensor::build(tensor_type_,"U_h",std::vector<size_t> (2, h_dim));
                    U_h("pq") = Feigen["eigenvectors"]("pq");
                    fill_eigen(block,h,Feigen["eigenvalues"].data());
                }
                ambit::Tensor U_out = U.block(block);
                combine_tensor(U_out,U_h,space,h);
            }
        }
    }
    return {eigenvalues_a, eigenvalues_b};
}

ambit::Tensor MRDSRG::separate_tensor(ambit::Tensor& tens, const Dimension& irrep, const int& h){
    // test tens and irrep
    int tens_dim = static_cast<int>(tens.dim(0));
    if(tens_dim != irrep.sum() || tens_dim != tens.dim(1)){
        throw PSIEXCEPTION("Wrong dimension for the to-be-separated ambit Tensor.");
    }
    if(h >= irrep.n()){
        throw PSIEXCEPTION("Ask for wrong irrep.");
    }

    // from relative (blocks) to absolute (big tensor) index
    auto rel_to_abs = [&](size_t i, size_t j, size_t offset){
        return (i + offset) * tens_dim + (j + offset);
    };

    // compute offset
    size_t offset = 0, h_dim = irrep[h];
    int h_local = h;
    while((--h_local) >= 0) offset += irrep[h_local];

    // fill in values
    ambit::Tensor T_h = ambit::Tensor::build(tensor_type_,"T_h",std::vector<size_t> (2, h_dim));
    for(size_t i = 0; i < h_dim; ++i){
        for(size_t j = 0; j < h_dim; ++j){
            size_t abs_idx = rel_to_abs(i, j, offset);
            T_h.data()[i * h_dim + j] = tens.data()[abs_idx];
        }
    }

    return T_h;
}

void MRDSRG::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep, const int& h){
    // test tens and irrep
    if(h >= irrep.n()){
        throw PSIEXCEPTION("Ask for wrong irrep.");
    }
    size_t tens_h_dim = tens_h.dim(0), h_dim = irrep[h];
    if(tens_h_dim != h_dim || tens_h_dim != tens_h.dim(1)){
        throw PSIEXCEPTION("Wrong dimension for the to-be-combined ambit Tensor.");
    }

    // from relative (blocks) to absolute (big tensor) index
    size_t tens_dim = tens.dim(0);
    auto rel_to_abs = [&](size_t i, size_t j, size_t offset){
        return (i + offset) * tens_dim + (j + offset);
    };

    // compute offset
    size_t offset = 0;
    int h_local = h;
    while((--h_local) >= 0) offset += irrep[h_local];

    // fill in values
    for(size_t i = 0; i < h_dim; ++i){
        for(size_t j = 0; j < h_dim; ++j){
            size_t abs_idx = rel_to_abs(i, j, offset);
            tens.data()[abs_idx] = tens_h.data()[i * h_dim + j];
        }
    }
}

void MRDSRG::print_cumulant_summary(){
    print_h2("Density Cumulant Summary");
    check_density(Lambda2_, "2-body");
    if(options_.get_str("THREEPDC") != "ZERO"){
        check_density(Lambda3_, "3-body");
    }
}

void MRDSRG::check_density(BlockedTensor& D, const std::string& name){
    int rank_half = D.rank() / 2;
    std::vector<std::string> labels;
    std::vector<double> maxes, norms;
    std::vector<std::string> blocks = D.block_labels();
    for(const auto& block: blocks){
        std::string spin_label;
        std::vector<int> idx;
        for(int i = 0; i < rank_half; ++i){
            idx.emplace_back(i);
        }
        for(const auto& i: idx){
            if(islower(block[i])){
                spin_label += "A";
            }else{
                spin_label += "B";
            }
        }
        labels.emplace_back(spin_label);

        double D_norm = 0.0, D_max = 0.0;
        D.block(block).citerate([&](const std::vector<size_t>&, const double& value){
            double abs_value = fabs(value);
            if(abs_value > 1.0e-15){
                if(abs_value > D_max) D_max = value;
                D_norm += value * value;
            }
        });
        maxes.emplace_back(D_max);
        norms.emplace_back(std::sqrt(D_norm));
    }

    int n = labels.size();
    std::string sep(10 + 13 * n, '-');
    std::string indent = "\n    ";
    std::string output = indent + str(boost::format("%-10s") % name);
    for(int i = 0; i < n; ++i)
        output += str(boost::format(" %12s") % labels[i]);
    output += indent + sep;

    output += indent + str(boost::format("%-10s") % "max");
    for(int i = 0; i < n; ++i)
        output += str(boost::format(" %12.6f") % maxes[i]);
    output += indent + str(boost::format("%-10s") % "norm");
    for(int i = 0; i < n; ++i)
        output += str(boost::format(" %12.6f") % norms[i]);
    output += indent + sep;
    outfile->Printf("%s", output.c_str());
}

}}

    #include <cmath>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>

#include "explorer.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

Explorer::Explorer(Options &options,ExplorerIntegrals* ints)
    : min_energy_(0.0),ints_(ints),pt2_energy_correction_(0.0)
{
    // Read data and allocate member objects
    startup(options);

    // Explore the space of excited configurations
    explore(options);

    // Optionally diagonalize a small Hamiltonian
    if(options.get_bool("COMPUTE_ENERGY")){
        if(options.get_str("ENERGY_TYPE") == "SELECT"){
            diagonalize_selected_space(options);
        }else
        if(options.get_str("ENERGY_TYPE") == "FULL"){
            diagonalize_p_space(options);
        }else
        if(options.get_str("ENERGY_TYPE") == "LOWDIN"){
            diagonalize_p_space_lowdin(options);
        }
    }
}

Explorer::~Explorer()
{
}

void Explorer::startup(Options& options)
{
    read_info(options);

    screen_mos();

    // Connect the integrals to the determinant class
    StringDeterminant::set_ints(ints_);

    // Build the reference determinant and compute its energy
    std::vector<int> occupation(2 * nmo_,0);
    int cumidx = 0;
    for (int h = 0; h < nirrep_; ++h){
        for (int i = 0; i < nalphapi_ref_[h]; ++i){
            occupation[i + cumidx] = 1;
        }
        for (int i = 0; i < nbetapi_ref_[h]; ++i){
            occupation[nmo_ + i + cumidx] = 1;
        }
        cumidx += nmopi_[h];
    }
    reference_determinant_ = StringDeterminant(occupation);

    min_energy_ = ref_energy_ = reference_determinant_.energy() + nuclear_repulsion_energy_;
    min_energy_determinant_ = reference_determinant_;
    fprintf(outfile,"\n  The tentative reference determinant is:");
    reference_determinant_.print();
    fprintf(outfile,"\n  and its energy: %.12f Eh",min_energy_);

    max_energy_ = min_energy_;

    ints_->make_fock_matrix(reference_determinant_.get_alfa_bits(),reference_determinant_.get_beta_bits());
}

void Explorer::read_info(Options& options)
{
    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    epsilon_a_ = wfn->epsilon_a();
    epsilon_b_ = wfn->epsilon_a();
    nmo_ = wfn->nmo();
    nmopi_ = wfn->nmopi();
    nalphapi_ = wfn->nalphapi();
    nbetapi_ = wfn->nbetapi();
    nalpha_ = wfn->nalpha();
    nbeta_ = wfn->nbeta();
    nirrep_ = wfn->nirrep();
    frzcpi_ = Dimension(nirrep_);
    frzvpi_ = Dimension(nirrep_);

    if (options["FROZEN_DOCC"].has_changed()){
        if (options["FROZEN_DOCC"].size() == nirrep_){
            for (int h = 0; h < nirrep_; ++h){
                frzcpi_[h] = options["FROZEN_DOCC"][h].to_integer();
            }
        }else{
            fprintf(outfile,"\n\n  The input array FROZEN_DOCC has information for %d irreps, this does not match the total number of irreps %d",
                    options["FROZEN_DOCC"].size(),nirrep_);
            fprintf(outfile,"\n  Exiting the program.\n");
            printf("  The input array FROZEN_DOCC has information for %d irreps, this does not match the total number of irreps %d",
                    options["FROZEN_DOCC"].size(),nirrep_);
            printf("\n  Exiting the program.\n");

            exit(Failure);
        }
    }

    if (options["FROZEN_UOCC"].has_changed()){
        if (options["FROZEN_UOCC"].size() == nirrep_){
            for (int h = 0; h < nirrep_; ++h){
                frzvpi_[h] = options["FROZEN_UOCC"][h].to_integer();
            }
        }else{
            fprintf(outfile,"\n\n  The input array FROZEN_UOCC has information for %d irreps, this does not match the total number of irreps %d",
                    options["FROZEN_UOCC"].size(),nirrep_);
            fprintf(outfile,"\n  Exiting the program.\n");
            printf("  The input array FROZEN_UOCC has information for %d irreps, this does not match the total number of irreps %d",
                    options["FROZEN_UOCC"].size(),nirrep_);
            printf("\n  Exiting the program.\n");

            exit(Failure);
        }
    } else if (options["ACTIVE"].has_changed()){
        if (options["ACTIVE"].size() == nirrep_){
            for (int h = 0; h < nirrep_; ++h){
                frzvpi_[h] = nmopi_[h] - frzcpi_[h] - options["ACTIVE"][h].to_integer();
            }
        }else{
            fprintf(outfile,"\n\n  The input array ACTIVE has information for %d irreps, this does not match the total number of irreps %d",
                    options["ACTIVE"].size(),nirrep_);
            fprintf(outfile,"\n  Exiting the program.\n");
            printf("  The input array ACTIVE has information for %d irreps, this does not match the total number of irreps %d",
                    options["ACTIVE"].size(),nirrep_);
            printf("\n  Exiting the program.\n");

            exit(Failure);
        }
    }

    // Create the vectors of frozen orbitals (in the Pitzer ordering)
    for (int h = 0, p = 0; h < nirrep_; ++h){
        for (int i = 0; i < frzcpi_[h]; ++i){
            frzc_.push_back(p + i);
        }
        p += nmopi_[h];
        for (int i = 0; i < frzvpi_[h]; ++i){
            frzv_.push_back(p - frzvpi_[h] + i);
        }
    }

    // Create the array with mo symmetry
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p){
            mo_symmetry_.push_back(h);
        }
    }

    int charge       = Process::environment.molecule()->molecular_charge();
    int multiplicity = Process::environment.molecule()->multiplicity();
    int nel = nalpha_ + nbeta_;

    // If the charge has changed recompute the number of electrons
    if(options["CHARGE"].has_changed()){
        charge = options.get_int("CHARGE");
        nel = 0;
        int natom = Process::environment.molecule()->natom();
        for(int i=0; i < natom;i++){
            nel += static_cast<int>(Process::environment.molecule()->Z(i));
        }
        nel -= charge;
    }

    if(options["MULTIPLICITY"].has_changed()){
        multiplicity = options.get_int("MULTIPLICITY");
    }

    if( ((nel + 1 - multiplicity) % 2) != 0)
        throw PSIEXCEPTION("\n\n  MOInfoBase: Wrong multiplicity.\n\n");
    nalpha_ = (nel + multiplicity - 1) / 2;
    nbeta_ =  nel - nalpha_;

    wavefunction_symmetry_ = 0;
    if(options["SYMMETRY"].has_changed()){
        wavefunction_symmetry_ = options.get_int("SYMMETRY");
    }

    boost::shared_ptr<Molecule> molecule_ = wfn->molecule();
    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    determinant_threshold_ = options.get_double("DET_THRESHOLD");
    if (options["DEN_THRESHOLD"].has_changed()){
        denominator_threshold_ = options.get_double("DEN_THRESHOLD");
    }else{
        denominator_threshold_ = 2.0 * determinant_threshold_;
    }
    space_m_threshold_ = options.get_double("SPACE_M_THRESHOLD");
    space_i_threshold_ = options.get_double("SPACE_I_THRESHOLD");
    if (space_m_threshold_ > determinant_threshold_){
        space_m_threshold_ = determinant_threshold_;
        space_i_threshold_ = determinant_threshold_;
        fprintf(outfile,"\n  The model space comprises all the determinants.\n  Modifying the model and intermediate space thresholds.\n");
    }
    if (space_m_threshold_ > space_i_threshold_){
        space_i_threshold_ = space_m_threshold_;
    }
    if (space_i_threshold_ > determinant_threshold_){
        space_i_threshold_ = determinant_threshold_;
        fprintf(outfile,"\n  Changing the value of the intermediate space threshold.\n");
    }

    t2_threshold_ = options.get_double("T2_THRESHOLD");

    if (options.get_str("SCREENING_TYPE") == "MP"){
        mp_screening_ = true;
    }else{
        mp_screening_ = false;
    }

    fprintf(outfile,"\n  Determinant threshold        = %.3f (Eh)",determinant_threshold_);
    fprintf(outfile,"\n  Denominator threshold        = %.3f (Eh)",denominator_threshold_);
    fprintf(outfile,"\n  Model space threshold        = %.3f (Eh)",space_m_threshold_);
    fprintf(outfile,"\n  Intermediate space threshold = %.3f (Eh)",space_i_threshold_);
    fprintf(outfile,"\n  Coupling threshold           = %.3f (muEh)",t2_threshold_ * 1000000.0);

    fprintf(outfile,"\n  String screening: %s (%s)",mp_screening_ ? "Moller-Plesset denominators" : "excited determinants",options.get_str("SCREENING_TYPE").c_str());
}

bool compare_tuples (const boost::tuple<double,int,int>& t1, const boost::tuple<double,int,int>& t2)
{
    if (t1.get<0>() != t2.get<0>()){
        return (t1.get<0>() < t2.get<0>());
    }
    else if (t1.get<1>() != t2.get<1>()){
        return (t1.get<1>() < t2.get<1>());
    }
    return (t1.get<2>() < t2.get<2>());
}

void Explorer::screen_mos()
{
    // Determine the best occupation using the orbital energies
    std::vector<boost::tuple<double,int,int> > sorted_ea;
    std::vector<boost::tuple<double,int,int> > sorted_eb;
    for (int h = 0, sump = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p, ++sump){
            sorted_ea.push_back(boost::make_tuple(epsilon_a_->get(h,p),h,sump));
            sorted_eb.push_back(boost::make_tuple(epsilon_b_->get(h,p),h,sump));
        }
    }

    std::sort(sorted_ea.begin(),sorted_ea.end(),compare_tuples);
    std::sort(sorted_eb.begin(),sorted_eb.end(),compare_tuples);

    nalphapi_ref_ = Dimension(nirrep_);
    nbetapi_ref_ = Dimension(nirrep_);
    minalphapi_ = Dimension(nirrep_);
    minbetapi_ = Dimension(nirrep_);
    maxalphapi_ = Dimension(nirrep_);
    maxbetapi_ = Dimension(nirrep_);

    fprintf(outfile,"\n\n                Molecular orbitals:");
    fprintf(outfile,"\n  ====================================================");
    fprintf(outfile,"\n     MO         alpha                  beta");
    fprintf(outfile,"\n           irrep    energy  occ   irrep    energy  occ");
    fprintf(outfile,"\n  ----------------------------------------------------");
    for (int p = 0; p < nmo_; ++p){
        double ea = sorted_ea[p].get<0>();
        double eb = sorted_eb[p].get<0>();
        int ha = sorted_ea[p].get<1>();
        int hb = sorted_eb[p].get<1>();
        int pa = sorted_ea[p].get<2>();
        int pb = sorted_eb[p].get<2>();
        mo_symmetry_qt_.push_back(ha);

//        double ea = std::get<0>(sorted_ea[p]);
//        double eb = std::get<0>(sorted_eb[p]);
//        int ha = std::get<1>(sorted_ea[p]);
//        int hb = std::get<1>(sorted_eb[p]);
//        int pa = std::get<2>(sorted_ea[p]);
//        int pb = std::get<2>(sorted_eb[p]);
        //if (std::max(std::fabs(ea),std::fabs(eb)) < denominator_threshold_ * 1.25)
        bool frozen = false;
        fprintf(outfile,"\n %6d    %3d %12.6f  %1d    %3d %12.6f  %1d",p,ha,ea,p < nalpha_,hb,eb,p < nbeta_);
        if (std::find(frzc_.begin(), frzc_.end(), pa) != frzc_.end()){
            frozen = true;
            fprintf(outfile," <- frozen");
        }
        if (std::find(frzv_.begin(), frzv_.end(), pa) != frzv_.end()){
            fprintf(outfile," <- frozen");
            frozen = true;
        }
        if (not frozen){
            epsilon_a_qt_.push_back(ea);
            epsilon_b_qt_.push_back(eb);
            qt_to_pitzer_.push_back(pa);
        }
    }
    fprintf(outfile,"\n  ----------------------------------------------------");

    for (int p = 0; p < nalpha_; ++p) nalphapi_ref_[sorted_ea[p].get<1>()] += 1;
    for (int p = 0; p < nbeta_; ++p) nbetapi_ref_[sorted_eb[p].get<1>()] += 1;
//    for (int p = 0; p < nalpha_; ++p) nalphapi_ref_[std::get<1>(sorted_ea[p])] += 1;
//    for (int p = 0; p < nbeta_; ++p) nbetapi_ref_[ std::get<1>(sorted_eb[p])] += 1;

    fprintf(outfile,"\n  Occupation numbers of the refence determinant:");
    fprintf(outfile,"|");
    for (int h = 0; h < nirrep_; ++h){
        fprintf(outfile," %d",nalphapi_ref_[h]);
    }
    fprintf(outfile," > x ");
    fprintf(outfile,"|");
    for (int h = 0; h < nirrep_; ++h){
        fprintf(outfile," %d",nbetapi_ref_[h]);
    }
    fprintf(outfile," >");

    double e_ahomo = sorted_ea[nalpha_ - 1].get<0>();
    double e_bhomo = sorted_eb[nbeta_ - 1].get<0>();
    double e_alumo = sorted_ea[nalpha_].get<0>();
    double e_blumo = sorted_eb[nbeta_].get<0>();

//    double e_ahomo = std::get<0>(sorted_ea[nalpha_ - 1]);
//    double e_bhomo = std::get<0>(sorted_eb[nbeta_ - 1]);
//    double e_alumo = std::get<0>(sorted_ea[nalpha_]);
//    double e_blumo = std::get<0>(sorted_eb[nbeta_]);

    fprintf(outfile,"\n  Energy of the alpha/beta HOMO: %12.6f %12.6f",e_ahomo,e_bhomo);
    fprintf(outfile,"\n  Energy of the alpha/beta LUMO: %12.6f %12.6f",e_alumo,e_blumo);
    // Determine the range of MOs to consider
    for (int h = 0; h < nirrep_; ++h){
        for (int p = nmopi_[h] - 1; p >=0 ; --p){
            if (e_alumo - epsilon_a_->get(h,p) < denominator_threshold_){
                minalphapi_[h] = p;
            }
            if (e_blumo - epsilon_b_->get(h,p) < denominator_threshold_){
                minbetapi_[h] = p;
            }
        }
        for (int p = 0; p < nmopi_[h]; ++p){
            if (epsilon_a_->get(h,p) - e_ahomo < denominator_threshold_){
                maxalphapi_[h] = p + 1;
            }
            if (epsilon_b_->get(h,p) - e_bhomo < denominator_threshold_){
                maxbetapi_[h] = p + 1;
            }
        }
    }

    fprintf(outfile,"\n  Orbital ranges:");
    fprintf(outfile,"|");
    for (int h = 0; h < nirrep_; ++h){
        fprintf(outfile," %d/%d",minalphapi_[h],maxalphapi_[h]);
    }
    fprintf(outfile," > x ");
    fprintf(outfile,"|");
    for (int h = 0; h < nirrep_; ++h){
        fprintf(outfile," %d/%d",minbetapi_[h],maxbetapi_[h]);
    }
    fprintf(outfile," >");
}




}} // EndNamespaces

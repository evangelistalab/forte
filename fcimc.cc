#include <cmath>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>

#include <libqt/qt.h>

#include "fcimc.h"

using namespace psi;

namespace psi{ namespace libadaptive{

std::default_random_engine generator_;
std::uniform_real_distribution<double> distribution_real_(0.0,1.0);
std::uniform_int_distribution<size_t> distribution_int_;
auto rand_real = std::bind ( distribution_real_, generator_);
auto rand_int = std::bind ( distribution_int_, generator_);

std::pair<size_t,size_t> generate_ind_random_pair(size_t range)
{
    size_t r1 = rand_int() % range;
    size_t r2 = rand_int() % (range - 1);
    if (r2 >= r1) r2++;
    return std::make_pair(r1,r2);
}

#ifdef _OPENMP
   #include <omp.h>
   bool FCIQMC::have_omp_ = true;
#else
   #define omp_get_max_threads() 1
   #define omp_get_thread_num() 0
   bool FCIQMC::have_omp_ = false;
#endif

FCIQMC::FCIQMC(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), ints_(ints)
{
    copy(wfn);
    startup();
}

FCIQMC::~FCIQMC()
{
}

void FCIQMC::startup()
{
    // Connect the integrals to the determinant class
    BitsetDeterminant::set_ints(ints_);

    // The number of correlated molecular orbitals
    ncmo_ = ints_->ncmo();
    ncmopi_ = ints_->ncmopi();

    // Overwrite the frozen orbitals arrays
    frzcpi_ = ints_->frzcpi();
    frzvpi_ = ints_->frzvpi();



    // Create the array with mo symmetry
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < ncmopi_[h]; ++p){
            mo_symmetry_.push_back(h);
        }
    }

    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    wavefunction_symmetry_ = 0;
    if(options_["ROOT_SYM"].has_changed()){
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }

    // Build the reference determinant and compute its energy
    std::vector<int> occupation(2 * ncmo_,0);
    int cumidx = 0;
    for (int h = 0; h < nirrep_; ++h){
        for (int i = 0; i < doccpi_[h] - frzcpi_[h]; ++i){
            occupation[i + cumidx] = 1;
            occupation[ncmo_ + i + cumidx] = 1;
        }
        for (int i = 0; i < soccpi_[h]; ++i){
            occupation[i + cumidx] = 1;
        }
        cumidx += ncmopi_[h];
    }
    BitsetDeterminant reference_determinant(occupation);

    outfile->Printf("\n  The reference determinant is:\n");
    reference_determinant.print();

    // Read options
//    nroot_ = options_.get_int("NROOT");
//    spawning_threshold_ = options_.get_double("SPAWNING_THRESHOLD");
//    initial_guess_spawning_threshold_ = options_.get_double("GUESS_SPAWNING_THRESHOLD");
    time_step_ = options_.get_double("TAU");
    maxiter_ = options_.get_int("MAXITER");
    start_num_det_ = options_.get_double("START_NUM_DET");
//    e_convergence_ = options_.get_double("E_CONVERGENCE");
//    energy_estimate_threshold_ = options_.get_double("ENERGY_ESTIMATE_THRESHOLD");

//    energy_estimate_freq_ = options_.get_int("ENERGY_ESTIMATE_FREQ");

//    adaptive_beta_ = options_.get_bool("ADAPTIVE_BETA");
//    fast_variational_estimate_ = options_.get_bool("FAST_EVAR");
    do_shift_ = options_.get_bool("USE_SHIFT");
//    do_simple_prescreening_ = options_.get_bool("SIMPLE_PRESCREENING");
//    do_dynamic_prescreening_ = options_.get_bool("DYNAMIC_PRESCREENING");

//    if (options_.get_str("PROPAGATOR") == "LINEAR"){
//        propagator_ = LinearPropagator;
//        propagator_description_ = "Linear";
//    }else if (options_.get_str("PROPAGATOR") == "QUADRATIC"){
//        propagator_ = QuadraticPropagator;
//        propagator_description_ = "Quadratic";
//    }else if (options_.get_str("PROPAGATOR") == "CUBIC"){
//        propagator_ = CubicPropagator;
//        propagator_description_ = "Cubic";
//    }else if (options_.get_str("PROPAGATOR") == "QUARTIC"){
//        propagator_ = QuarticPropagator;
//        propagator_description_ = "Quartic";
//    }else if (options_.get_str("PROPAGATOR") == "POWER"){
//        propagator_ = PowerPropagator;
//        propagator_description_ = "Power";
//    }else if (options_.get_str("PROPAGATOR") == "POSITIVE"){
//        propagator_ = PositivePropagator;
//        propagator_description_ = "Positive";
//    }
    shift_ = 0.0;
    nWalkers_ = 0;
    num_threads_ = omp_get_max_threads();
}

void FCIQMC::print_info()
{
    // Print a summary
    std::vector<std::pair<std::string,int>> calculation_info{
        {"Symmetry",wavefunction_symmetry_},
//        {"Number of roots",nroot_},
//        {"Root used for properties",options_.get_int("ROOT")},
        {"Maximum number of steps",maxiter_},
//        {"Energy estimation frequency",energy_estimate_freq_},
        {"Number of threads",num_threads_}};

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"Time step (beta)",time_step_},
//        {"Spawning threshold",spawning_threshold_},
//        {"Initial guess spawning threshold",initial_guess_spawning_threshold_},
//        {"Convergence threshold",e_convergence_},
//        {"Prescreening tollerance factor",prescreening_tollerance_factor_},
//        {"Energy estimate tollerance",energy_estimate_threshold_}
    };

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
//        {"Propagator type",propagator_description_},
//        {"Adaptive time step",adaptive_beta_ ? "YES" : "NO"},
//        {"Shift the energy",do_shift_ ? "YES" : "NO"},
//        {"Prescreen spawning",do_simple_prescreening_ ? "YES" : "NO"},
//        {"Dynamic prescreening",do_dynamic_prescreening_ ? "YES" : "NO"},
//        {"Fast variational estimate",fast_variational_estimate_ ? "YES" : "NO"},
        {"Using OpenMP", have_omp_ ? "YES" : "NO"},
    };

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-39s %10d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-39s %10.3e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-39s %10s",str_dim.first.c_str(),str_dim.second.c_str());
    }
    outfile->Flush();
}


double FCIQMC::compute_energy()
{
    timer_on("FCIQMC:Energy");
    outfile->Printf("\n\n\t  ---------------------------------------------------------");
    outfile->Printf("\n\t      Adaptive Path-Integral Full Configuration Interaction");
    outfile->Printf("\n\t         by Tianyuan Zhang and Francesco A. Evangelista");
    outfile->Printf("\n\t                    %4d thread(s) %s",num_threads_,have_omp_ ? "(OMP)" : "");
    outfile->Printf("\n\t  ---------------------------------------------------------");

    // Print a summary of the options
    print_info();

    // Build the reference determinant and compute its energy
    std::vector<bool> occupation_a(nmo_,false);
    std::vector<bool> occupation_b(nmo_,false);
    int cumidx = 0;
    for (int h = 0; h < nirrep_; ++h){
        for (int i = 0; i < doccpi_[h]; ++i){
            occupation_a[i + cumidx] = true;
            occupation_b[i + cumidx] = true;
        }
        for (int i = 0; i < soccpi_[h]; ++i){
            occupation_a[i + doccpi_[h] + cumidx] = true;
        }
        cumidx += nmopi_[h];
    }
    BitsetDeterminant reference(occupation_a,occupation_b);
    reference.print();

    Ehf_ = reference.slater_rules(reference);

    double nre = molecule_->nuclear_repulsion_energy();
    outfile->Printf("\nnuclear_repulsion_energy:%lf, Ehf:%lf",nre, Ehf_);

    // Create the initial walker population
    std::map<BitsetDeterminant,double> walkers;
    walkers[reference] = start_num_det_;

    for (size_t iter = 0; iter < maxiter_; ++iter){

        std::map<BitsetDeterminant,double> new_walkers;

        // Step #1.  Spawning
        timer_on("FCIQMC:Spawn");
        spawn(walkers,new_walkers);
        timer_off("FCIQMC:Spawn");
        outfile->Printf("\nRef walkers: %f after Spawn", new_walkers[reference]);

        // Step #2.  Death/Clone
        timer_on("FCIQMC:Death_Clone");
        death_clone(walkers, shift_);
        timer_off("FCIQMC:Death_Clone");

        outfile->Printf("\nRef walkers: %f after Death/Clone", new_walkers[reference]);

        // Step #3.  Merge parents and spawned
        timer_on("FCIQMC:Merge");
        merge(walkers, new_walkers);
        timer_off("FCIQMC:Merge");

        outfile->Printf("\nRef walkers: %f after merge", new_walkers[reference]);


        // Step #3.  annihilation
        timer_on("FCIQMC:Annihilation");
        annihilate(walkers,new_walkers,spawning_threshold_);
        timer_off("FCIQMC:Annihilation");

        print_iter_info(iter,reference, walkers, true, true, true);
    }


//    double energy1 = reference.energy() + nre;
//    double energy2 = reference.slater_rules(reference) + nre;
//    outfile->Printf("\n  Energy 1: %f",energy1);
//    outfile->Printf("\n  Energy 2: %f",energy2);

//    excited.set_alfa_bit(1,false);
//    excited.set_beta_bit(1,false);:
//    excited.set_alfa_bit(6,true);
//    excited.set_beta_bit(6,true);

//    excited.print();

//    std::vector<BitsetDeterminant> walkers;
//    walkers.push_back(reference);
//    walkers.push_back(excited);
//    size_t nwalkers = walkers.size();

//    Matrix H("Hamiltonian",nwalkers,nwalkers);
//    Matrix evecs("Eigenvectors",nwalkers,nwalkers);
//    Vector evals("Eigenvalues",nwalkers);

//    for (size_t I = 0; I < nwalkers; ++I){
//        for (size_t J = 0; J < nwalkers; ++J){
//            double HIJ = walkers[I].slater_rules(walkers[J]) + (I == J ? nre : 0.0);
//            H.set(I,J,HIJ);
//        }
//    }
//    H.print();

//    H.diagonalize(evecs,evals);
//    evecs.print();
//    evals.print();

    timer_off("FCIQMC:Energy");
    return 0.0;
}


void FCIQMC::spawn(walker_map& walkers,walker_map& new_walkers)
{
    int count = 0;
    for (auto& det_coef : walkers){
        const BitsetDeterminant& det = det_coef.first;
        double coef = det_coef.second;
//        size_t nsa,nsb,ndaa,ndab,ndbb;
//        std::tuple<size_t,size_t,size_t,size_t,size_t> pgen = compute_pgen(det);
//        std::tie (nsa,nsb,ndaa,ndab,ndbb) = pgen;

//        size_t sumgen = nsa + nsb + ndaa + ndbb + ndab;
        timer_on("FCIQMC:Compute_excitations");
        std::vector<std::tuple<size_t,size_t>> singleExcitations;
        std::vector<std::tuple<size_t,size_t,size_t,size_t>> doubleExcitations;
        compute_excitations(det, singleExcitations, doubleExcitations);
        size_t sumSingle = singleExcitations.size(), sumDouble = doubleExcitations.size();
        size_t sumgen = sumSingle+sumDouble;
        timer_off("FCIQMC:Compute_excitations");

        size_t nid = std::round(std::fabs(coef));
        for (size_t detW = 0; detW < nid; ++detW){
            BitsetDeterminant new_det(det);
            // Select a random number within the range of allowed determinants
//            singleWalkerSpawn(new_det,det,pgen,sumgen);
            size_t rand_ext = rand_int() % sumgen;
            if (rand_ext<sumDouble)
                detDoubleExcitation(new_det, doubleExcitations[rand_ext]);
            else
                detSingleExcitation(new_det, singleExcitations[rand_ext-sumDouble]);


            double HIJ = new_det.slater_rules(det);
            double pspawn = time_step_ * std::fabs(HIJ) * double(sumgen);
            int pspawn_floor = std::floor(pspawn);
            if (rand_real() < pspawn - double(pspawn_floor)){
                pspawn_floor++;
            }

            int nspawn = coef * HIJ > 0 ? -pspawn_floor : pspawn_floor;

            // TODO: check
            if (nspawn != 0){
                new_walkers[new_det] += double(nspawn);
            }

//            outfile->Printf("\n  Determinant %d:",count);
//            det.print();
//            outfile->Printf(" spawned %d (%f):",nspawn,pspawn);
//            new_det.print();
//            if (count == 15){
//                outfile->Printf("\n  Determinant %d spawn detail:",count);
//                outfile->Printf("\n  Random: %zu",rand_ext);
//                outfile->Printf("\n  NS %zu, ND %zu",sumSingle,sumDouble);
//                size_t ii,aa,jj,bb;
//                std::tie (ii,aa,jj,bb) = doubleExcitations[rand_ext];
//                outfile->Printf("\n  Ext: %zu %zu %zu %zu",ii,aa,jj,bb);
//                std::vector<std::tuple<size_t,size_t>>::iterator t1 ;
//                for(t1=singleExcitations.begin(); t1!=singleExcitations.end(); t1++){
//                    size_t ii,aa;
//                    std::tie (ii,aa) = *t1;
//                    outfile->Printf("\n %zu %zu",ii,aa);
//                }
//                std::vector<std::tuple<size_t,size_t,size_t,size_t>>::iterator t2 ;
//                for(t2=doubleExcitations.begin(); t2!=doubleExcitations.end(); t2++){
//                    size_t ii,aa,jj,bb;
//                    std::tie (ii,aa,jj,bb) = *t2;
//                    outfile->Printf("\n %zu %zu %zu %zu",ii,aa,jj,bb);
//                }
//            }
        }
        count++;
    }

}

void FCIQMC::singleWalkerSpawn(BitsetDeterminant & new_det, const BitsetDeterminant& det, std::tuple<size_t,size_t,size_t,size_t,size_t> pgen, size_t sumgen)
{

    size_t nsa,nsb,ndaa,ndab,ndbb;
    std::tie (nsa,nsb,ndaa,ndab,ndbb) = pgen;
    size_t rand_class = rand_int() % sumgen;
    if (rand_class < ndab){
        const std::vector<int> aocc = det.get_alfa_occ();
        const std::vector<int> avir = det.get_alfa_vir();
        const std::vector<int> bocc = det.get_beta_occ();
        const std::vector<int> bvir = det.get_beta_vir();
        size_t i = aocc[rand_int() % aocc.size()];
        size_t j = bocc[rand_int() % bocc.size()];
        size_t a = avir[rand_int() % avir.size()];
        size_t b = bvir[rand_int() % bvir.size()];
        new_det.set_alfa_bit(i,false);
        new_det.set_alfa_bit(a,true);
        new_det.set_beta_bit(j,false);
        new_det.set_beta_bit(b,true);
    }else if (rand_class < ndab + ndaa){
        const std::vector<int> aocc = det.get_alfa_occ();
        const std::vector<int> avir = det.get_alfa_vir();
        std::pair<size_t,size_t> ij = generate_ind_random_pair(aocc.size());
        std::pair<size_t,size_t> ab = generate_ind_random_pair(avir.size());
        size_t i = aocc[ij.first];
        size_t j = aocc[ij.second];
        size_t a = avir[ab.first];
        size_t b = avir[ab.second];
        new_det.set_alfa_bit(i,false);
        new_det.set_alfa_bit(j,false);
        new_det.set_alfa_bit(a,true);
        new_det.set_alfa_bit(b,true);
    }else if (rand_class < ndab + ndaa + ndbb){
        const std::vector<int> bocc = det.get_beta_occ();
        const std::vector<int> bvir = det.get_beta_vir();
        std::pair<size_t,size_t> ij = generate_ind_random_pair(bocc.size());
        std::pair<size_t,size_t> ab = generate_ind_random_pair(bvir.size());
        size_t i = bocc[ij.first];
        size_t j = bocc[ij.second];
        size_t a = bvir[ab.first];
        size_t b = bvir[ab.second];
        new_det.set_beta_bit(i,false);
        new_det.set_beta_bit(j,false);
        new_det.set_beta_bit(a,true);
        new_det.set_beta_bit(b,true);
    }else if (rand_class < ndab + ndaa + ndbb + nsa){
        const std::vector<int> aocc = det.get_alfa_occ();
        const std::vector<int> avir = det.get_alfa_vir();
        size_t i = aocc[rand_int() % aocc.size()];
        size_t a = avir[rand_int() % avir.size()];
        new_det.set_alfa_bit(i,false);
        new_det.set_alfa_bit(a,true);
    }else{
        const std::vector<int> bocc = det.get_beta_occ();
        const std::vector<int> bvir = det.get_beta_vir();
        size_t i = bocc[rand_int() % bocc.size()];
        size_t a = bvir[rand_int() % bvir.size()];
        new_det.set_beta_bit(i,false);
        new_det.set_beta_bit(a,true);
    }
}

// Step #2.  Death/Clone
void FCIQMC::death_clone(walker_map& walkers, double shift)
{
    for (auto& det_coef : walkers){
        const BitsetDeterminant& det = det_coef.first;
        double coef = det_coef.second;
        double HII = det.energy();
        double pDeathClone = time_step_ * (HII-Ehf_-shift);

        int pDC_trunc = std::trunc(pDeathClone);

        size_t nid = std::round(std::fabs(coef));
        double signCoef = coef >= 0.0 ? 1.0 : -1.0;
        for (size_t detW=0; detW < nid; ++detW){

            if (rand_real() > std::fabs(pDeathClone-pDC_trunc)){
                coef -= signCoef * pDC_trunc;
            }else {
                coef -= signCoef * (pDC_trunc + (pDeathClone >= 0.0 ? 1.0 : -1.0));
            }
        }
//        det.print();
//        outfile->Printf("\nold coef=%lf, new coef=%lf, pDC=%lf",det_coef.second, coef, pDeathClone);

        walkers[det] = coef;

//        if (pDeathClone<0)
//            detClone(walkers, det, coef, pDeathClone);
//        else
//            detDeath(walkers, det, coef, pDeathClone);
    }

}

void FCIQMC::detClone(walker_map& walkers, const BitsetDeterminant& det, double coef, double pDeathClone){
    int pDC_trunc = std::trunc(pDeathClone);
    size_t nid = std::round(std::fabs(coef));
    int cloneCount = 0;
    for (size_t detW = 0; detW < nid; ++detW){
        if (rand_real() < double(pDC_trunc)-pDeathClone){
            cloneCount += pDC_trunc-1;
        } else {
            cloneCount += pDC_trunc;
        }

    }
    if (cloneCount<0){
        double signFlag = coef>0.0?-1.0:1.0;
        walkers[det] += signFlag*cloneCount;
//        outfile->Printf("\n  Determinant:");
//        det.print();
//        outfile->Printf("%f dets cloned %d:",nid,signFlag*cloneCount);
    }else{
//        outfile->Printf("\n  Determinant:");
//        det.print();
//        outfile->Printf("dets did not clone");
    }

}

void FCIQMC::detDeath(walker_map& walkers, const BitsetDeterminant& det, double coef, double pDeathClone){
    int pDC_trunc = std::trunc(pDeathClone);
    size_t nid = std::round(std::fabs(coef));
    int deathCount = 0;
    for (size_t detW = 0; detW < nid; ++detW){
        if (rand_real() < pDeathClone-double(pDC_trunc)){
            deathCount += pDC_trunc+1;
        } else {
            deathCount += pDC_trunc;
        }

    }
    if (deathCount<0){
        double signFlag = coef>0.0?-1.0:1.0;
        walkers[det] += signFlag*deathCount;
//        outfile->Printf("\n  Determinant:");
//        det.print();
//        outfile->Printf("%f dets died %d:",nid,signFlag*deathCount);
    }else{
//        outfile->Printf("\n  Determinant:");
//        det.print();
//        outfile->Printf("dets did not die");
    }

}

// Step #3.  Merge
void FCIQMC::merge(walker_map& walkers,walker_map& new_walkers){
    std::vector<BitsetDeterminant> removeDets;
    for (auto& det_coef : walkers){
        const BitsetDeterminant& det = det_coef.first;
        if (new_walkers.count(det)){
            walkers[det] += new_walkers[det];
            new_walkers.erase(det);
        }
        if (int(std::round(walkers[det]))==0){
            removeDets.push_back(det);
        }
    }
    for (auto& det_coef : new_walkers){
        const BitsetDeterminant& det = det_coef.first;
        walkers[det] = det_coef.second;
        if (int(std::round(walkers[det]))==0){
            removeDets.push_back(det);
        }
    }
    for (auto& det : removeDets) {
        walkers.erase(det);
    }
}

// Step #3.  Annihilation
void FCIQMC::annihilate(walker_map& walkers,walker_map& new_walkers,double spawning_threshold)
{

}

std::tuple<size_t,size_t,size_t,size_t,size_t> FCIQMC::compute_pgen(const BitsetDeterminant &det)
{
    const std::vector<int> aocc = det.get_alfa_occ();
    const std::vector<int> bocc = det.get_beta_occ();
    const std::vector<int> avir = det.get_alfa_vir();
    const std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    size_t nsa = 0;
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a){
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
                nsa++;
            }
        }
    }
    size_t nsb = 0;
    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a){
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
                nsb++;
            }
        }
    }

    size_t ndaa = 0;
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j){
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b){
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                        ndaa++;
                    }
                }
            }
        }
    }

    size_t ndab = 0;
    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j){
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b){
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                        ndab++;
                    }
                }
            }
        }
    }

    size_t ndbb = 0;
    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j){
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a){
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b){
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == wavefunction_symmetry_){
                        ndbb++;
                    }
                }
            }
        }
    }
    return std::make_tuple(nsa,nsb,ndaa,ndab,ndbb);
}

void FCIQMC::compute_excitations(const BitsetDeterminant &det, std::vector<std::tuple<size_t,size_t>>& singleExcitations, std::vector<std::tuple<size_t,size_t,size_t,size_t>>& doubleExcitations)
{
    compute_single_excitations(det, singleExcitations);
    compute_double_excitations(det, doubleExcitations);
}

void FCIQMC::compute_single_excitations(const BitsetDeterminant &det, std::vector<std::tuple<size_t,size_t>>& singleExcitations)
{
    const std::vector<int> aocc = det.get_alfa_occ();
    const std::vector<int> bocc = det.get_beta_occ();
    const std::vector<int> avir = det.get_alfa_vir();
    const std::vector<int> bvir = det.get_beta_vir();

//    det.print();

//    for (size_t ii : aocc){
//        outfile->Printf("\n  aocc = %d",ii);
//    }

//    for (size_t aa : avir){
//        outfile->Printf("\n  avir = %d",aa);
//    }
    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a){
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_){
                singleExcitations.push_back(std::make_tuple(ii,aa));
            }
        }
    }
    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a){
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == wavefunction_symmetry_){
                singleExcitations.push_back(std::make_tuple(ii+nmo_,aa+nmo_));
            }
        }
    }
}

void FCIQMC::compute_double_excitations(const BitsetDeterminant &det, std::vector<std::tuple<size_t,size_t,size_t,size_t>>& doubleExcitations)
{
    const std::vector<int> aocc = det.get_alfa_occ();
    const std::vector<int> bocc = det.get_beta_occ();
    const std::vector<int> avir = det.get_alfa_vir();
    const std::vector<int> bvir = det.get_beta_vir();

    int noalpha = aocc.size();
    int nobeta  = bocc.size();
    int nvalpha = avir.size();
    int nvbeta  = bvir.size();

    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j){
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b){
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                        doubleExcitations.push_back(std::make_tuple(ii,aa,jj,bb));
                    }
                }
            }
        }
    }


    for (int i = 0; i < noalpha; ++i){
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j){
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a){
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b){
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                        doubleExcitations.push_back(std::make_tuple(ii,aa,jj+nmo_,bb+nmo_));
                    }
                }
            }
        }
    }

    for (int i = 0; i < nobeta; ++i){
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j){
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a){
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b){
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == wavefunction_symmetry_){
                        doubleExcitations.push_back(std::make_tuple(ii+nmo_,aa+nmo_,jj+nmo_,bb+nmo_));
                    }
                }
            }
        }
    }
}

void FCIQMC::detSingleExcitation(BitsetDeterminant &new_det, std::tuple<size_t,size_t>& rand_ext){
    size_t ii,aa;
    std::tie (ii,aa) = rand_ext;
    if (ii<nmo_){
        new_det.set_alfa_bit(ii,false);
        new_det.set_alfa_bit(aa,true);
    } else {
        new_det.set_beta_bit(ii-nmo_,false);
        new_det.set_beta_bit(aa-nmo_,true);
    }

}

void FCIQMC::detDoubleExcitation(BitsetDeterminant &new_det, std::tuple<size_t,size_t,size_t,size_t>& rand_ext){
    size_t ii,aa,jj,bb;
    std::tie (ii,aa,jj,bb) = rand_ext;
    if (ii>=nmo_){
        new_det.set_beta_bit(ii-nmo_,false);
        new_det.set_beta_bit(jj-nmo_,false);
        new_det.set_beta_bit(aa-nmo_,true);
        new_det.set_beta_bit(bb-nmo_,true);
    }else if(jj>=nmo_){
        new_det.set_alfa_bit(ii,false);
        new_det.set_alfa_bit(aa,true);
        new_det.set_beta_bit(jj-nmo_,false);
        new_det.set_beta_bit(bb-nmo_,true);
    }else{
        new_det.set_alfa_bit(ii,false);
        new_det.set_alfa_bit(jj,false);
        new_det.set_alfa_bit(aa,true);
        new_det.set_alfa_bit(bb,true);
    }
}

double FCIQMC::count_walkers(walker_map& walkers) {
    double countWalkers = 0;
    for (auto walker:walkers){
        double Cwalker = walker.second;
        countWalkers+=std::fabs(Cwalker);
    }
    return countWalkers;
}

double FCIQMC::compute_proj_energy(BitsetDeterminant& ref, walker_map& walkers) {
    double Cref = walkers[ref];
    double Eproj = nuclear_repulsion_energy_;
    for (auto walker:walkers){
        const BitsetDeterminant& det = walker.first;
        double Cwalker = walker.second;
        Eproj += ref.slater_rules(det)*Cwalker/Cref;
    }
    return Eproj;
}

double FCIQMC::compute_var_energy(walker_map& walkers) {
    double Evar = 0;
    double mod2 = 0.0;
    for (auto walker:walkers){
        const BitsetDeterminant& det = walker.first;
        double Cwalker = walker.second;
        mod2 += Cwalker*Cwalker;
        for (auto walker2:walkers){
            const BitsetDeterminant& det2 = walker2.first;
            double Cwalker2 = walker2.second;
            Evar += det.slater_rules(det2)*Cwalker*Cwalker2;
        }
//        det.print();
//        outfile->Printf("\nCwalker: %lf",Cwalker);
    }
    Evar /= mod2;
    Evar += nuclear_repulsion_energy_;
    return Evar;
}

void FCIQMC::print_iter_info(size_t iter, BitsetDeterminant& ref, walker_map& walkers, bool countWalkers, bool calcEproj, bool calcEvar){
    outfile->Printf("\niter:%zu ended with %zu dets",iter, walkers.size());
    if (countWalkers)
        outfile->Printf(", %zu walkers", count_walkers(walkers));
    if (calcEproj)
        outfile->Printf(", proj E=%lf", compute_proj_energy(ref, walkers));
    if (calcEvar)
        outfile->Printf(", var E=%lf", compute_var_energy(walkers));
}

}} // EndNamespaces

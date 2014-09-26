#include <cmath>

#include <psifiles.h>
#include <libiwl/iwl.h>
#include <libtrans/integraltransform.h>
#include <libpsio/psio.hpp>
#include <libmints/matrix.h>

#include "integrals.h"
#include "memory.h"

using namespace std;
using namespace psi;

#include <psi4-dec.h>

namespace psi{ namespace libadaptive{

ExplorerIntegrals::ExplorerIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core)
    : options_(options), restricted_(restricted), resort_frozen_core_(resort_frozen_core), core_energy_(0.0), scalar_(0.0), ints_(nullptr)
{
    startup();
    transform_integrals();
    read_one_electron_integrals();
    read_two_electron_integrals();
    update_integrals();
}

ExplorerIntegrals::~ExplorerIntegrals()
{
    cleanup();
}

void ExplorerIntegrals::update_integrals()
{
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
    }
}

void ExplorerIntegrals::retransform_integrals()
{
    transform_integrals();
    read_one_electron_integrals();
    read_two_electron_integrals();
    update_integrals();
}

void ExplorerIntegrals::startup()
{
    // Grab the global (default) PSIO object, for file I/O
    boost::shared_ptr<PSIO> psio(_default_psio_lib_);

    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    if (not wfn){
        fprintf(outfile,"\n  No wave function object found!  Run a scf calculation first!\n");
        fflush(outfile);
        exit(1);
    }

    nirrep_ = wfn->nirrep();
    nso_ = wfn->nso();
    nmo_ = wfn->nmo();
    nmopi_ = wfn->nmopi();
    frzcpi_ = Dimension(nirrep_,"Frozen core orbitals per irrep");
    frzvpi_ = Dimension(nirrep_,"Frozen virtual orbitals per irrep");

    if (options_["FROZEN_DOCC"].has_changed() or options_["FROZEN_UOCC"].has_changed()){
        fprintf(outfile,"\n  Using the input to select the number of frozen core/virtual orbitals.\n");
        if (options_["FROZEN_DOCC"].size() == nirrep_){
            for (int h = 0; h < nirrep_; ++h){
                frzcpi_[h] = options_["FROZEN_DOCC"][h].to_integer();
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
        if (options_["FROZEN_UOCC"].size() == nirrep_){
            for (int h = 0; h < nirrep_; ++h){
                frzvpi_[h] = options_["FROZEN_UOCC"][h].to_integer();
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
    }else{
        fprintf(outfile,"\n  Using the Wavefunction object to select the number of frozen core/virtual orbitals.\n");
        frzcpi_ = wfn->frzcpi();
        frzvpi_ = wfn->frzvpi();
    }

    ncmopi_ = nmopi_;
    for (int h = 0; h < nirrep_; ++h){
        ncmopi_[h] -= frzcpi_[h] + frzvpi_[h];
    }
    ncmo_ = ncmopi_.sum();

    fprintf(outfile,"\n  ==> Integral Transformation <==\n");
    fprintf(outfile,"\n  Number of molecular orbitals:            %5d",nmopi_.sum());
    fprintf(outfile,"\n  Number of correlated molecular orbitals: %5zu",ncmo_);
    fprintf(outfile,"\n  Number of frozen occupied orbitals:      %5d",frzcpi_.sum());
    fprintf(outfile,"\n  Number of frozen unoccupied orbitals:    %5d\n\n",frzvpi_.sum());

    ncmo2_ = ncmo_ * ncmo_;
    ncmo3_ = ncmo_ * ncmo_ * ncmo_;

    num_oei = INDEX2(nmo_ - 1, nmo_ - 1) + 1;
    num_tei = INDEX4(nmo_ - 1,nmo_ - 1,nmo_ - 1,nmo_ - 1) + 1;
    num_aptei = nmo_ * nmo_ * nmo_ * nmo_;

    allocate();
}

void ExplorerIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals
    one_electron_integrals_a = new double[nmo_ * nmo_];
    one_electron_integrals_b = new double[nmo_ * nmo_];

    fock_matrix_a = new double[nmo_ * nmo_];
    fock_matrix_b = new double[nmo_ * nmo_];

    // Allocate the memory required to store the two-electron integrals
    aphys_tei_aa = new double[num_aptei];
    aphys_tei_ab = new double[num_aptei];
    aphys_tei_bb = new double[num_aptei];

    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];
}

void ExplorerIntegrals::deallocate()
{
    if (ints_ != nullptr) delete ints_;

    // Deallocate the memory required to store the one-electron integrals
    delete[] one_electron_integrals_a;
    delete[] one_electron_integrals_b;

    delete[] fock_matrix_a;
    delete[] fock_matrix_b;

    // Allocate the memory required to store the two-electron integrals
    delete[] aphys_tei_aa;
    delete[] aphys_tei_ab;
    delete[] aphys_tei_bb;

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;
}

void ExplorerIntegrals::cleanup()
{
    deallocate();
}

void ExplorerIntegrals::transform_integrals()
{
    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    // For now, we'll just transform for closed shells and generate all integrals.
    std::vector<boost::shared_ptr<MOSpace> > spaces;

    // TODO: transform only the orbitals within an energy range to save time on this step.
    spaces.push_back(MOSpace::all);

    // If the integral
    if (ints_ != nullptr) delete ints_;

    // Call IntegralTransform asking for integrals over restricted or unrestricted orbitals
    if (restricted_){
        ints_ = new IntegralTransform(wfn, spaces, IntegralTransform::Restricted, IntegralTransform::IWLOnly,IntegralTransform::PitzerOrder,IntegralTransform::None);
    }else{
        ints_ = new IntegralTransform(wfn, spaces, IntegralTransform::Unrestricted, IntegralTransform::IWLOnly,IntegralTransform::PitzerOrder,IntegralTransform::None);
    }

    // Keep the SO integrals on disk in case we want to retransform them
    ints_->set_keep_iwl_so_ints(true);
    ints_->transform_tei(MOSpace::all, MOSpace::all, MOSpace::all, MOSpace::all);
}

void ExplorerIntegrals::read_one_electron_integrals()
{
    for (size_t pq = 0; pq < nmo_ * nmo_; ++pq) one_electron_integrals_a[pq] = 0.0;
    for (size_t pq = 0; pq < nmo_ * nmo_; ++pq) one_electron_integrals_b[pq] = 0.0;
    double* packed_oei = new double[num_oei];

    if(restricted_){
        // Read the one-electron integrals (T + V, restricted)
        for (size_t pq = 0; pq < num_oei; ++pq) packed_oei[pq] = 0.0;
        iwl_rdone(PSIF_OEI,PSIF_MO_OEI,packed_oei,num_oei,0,0,outfile);
        for (int p = 0; p < nmo_; ++p){
            for (int q = p; q < nmo_; ++q){
                one_electron_integrals_a[p * nmo_ + q] = one_electron_integrals_a[q * nmo_ + p] = packed_oei[p + ioff[q]];
                one_electron_integrals_b[p * nmo_ + q] = one_electron_integrals_b[q * nmo_ + p] = packed_oei[p + ioff[q]];
            }
        }
    }else{
        // Read the alpha-alpha one-electron integrals (T + V)
        for (size_t pq = 0; pq < num_oei; ++pq) packed_oei[pq] = 0.0;
        iwl_rdone(PSIF_OEI,PSIF_MO_A_OEI,packed_oei,num_oei,0,0,outfile);
        for (int p = 0; p < nmo_; ++p){
            for (int q = p; q < nmo_; ++q){
                one_electron_integrals_a[p * nmo_ + q] = one_electron_integrals_a[q * nmo_ + p] = packed_oei[p + ioff[q]];
            }
        }
        for (size_t pq = 0; pq < num_oei; ++pq) packed_oei[pq] = 0.0;
        iwl_rdone(PSIF_OEI,PSIF_MO_B_OEI,packed_oei,num_oei,0,0,outfile);
        for (int p = 0; p < nmo_; ++p){
            for (int q = p; q < nmo_; ++q){
                one_electron_integrals_b[p * nmo_ + q] = one_electron_integrals_b[q * nmo_ + p] = packed_oei[p + ioff[q]];
            }
        }
    }
    delete[] packed_oei;

//    int num_oei_so = INDEX2(nso_ - 1, nso_ - 1) + 1;
//    for (size_t p = 0; p < nmo_; ++p) diagonal_kinetic_energy_integrals[p] = 0.0;
//    double* packed_oei_so = new double[num_oei_so];

//    // Read the kinetic energy integrals (restricted integrals, T)
//    for (size_t pq = 0; pq < num_oei_so; ++pq) packed_oei_so[pq] = 0.0;
//    iwl_rdone(PSIF_OEI,PSIF_SO_T,packed_oei_so,num_oei_so,0,0,outfile);

//    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
//    SharedMatrix Ca = wfn->Ca();
//    Dimension mopi_ = Ca->colspi();
//    Dimension sopi = Ca->rowspi();

//    // Transform the SO integrals to the MO basis
//    int q = 0;
//    int rho = 0;
//    for (int h = 0; h < nirrep_; ++h){
//        double** c = Ca->pointer(h);
//        for (int p = 0; p < mopi_[h]; ++p){
//            double t_int = 0.0;
//            for (int mu = 0; mu < sopi[h]; ++mu){
//                for (int nu = 0; nu < sopi[h]; ++nu){
//                    t_int += packed_oei_so[INDEX2(rho + mu,rho + nu)] * c[mu][p] * c[nu][p];
//                }
//            }
//            diagonal_kinetic_energy_integrals[q] = t_int;
//            q++;
//        }
//        rho += sopi[h];
//    }
//    delete[] packed_oei_so;
}

void ExplorerIntegrals::read_two_electron_integrals()
{
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_aa[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_ab[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_bb[pqrs] = 0.0;

    int ioffmax = 30000;
    int* myioff = new int[ioffmax];
    myioff[0] = 0;
    for(int i = 1; i < ioffmax; ++i)
        myioff[i] = myioff[i-1] + i;

    if (restricted_){
        double* two_electron_integrals = new double[num_tei];
        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries       
        for (size_t pqrs = 0; pqrs < num_tei; ++pqrs) two_electron_integrals[pqrs] = 0.0;

        // Read the integrals
        struct iwlbuf V_AAAA;
        iwl_buf_init(&V_AAAA,PSIF_MO_TEI, 0.0, 1, 1);
        iwl_buf_rd_all(&V_AAAA, two_electron_integrals, myioff, myioff, 0, myioff, 0, outfile);
        iwl_buf_close(&V_AAAA, 1);

        // Store the integrals
        for (size_t p = 0; p < nmo_; ++p){
            for (size_t q = 0; q < nmo_; ++q){
                for (size_t r = 0; r < nmo_; ++r){
                    for (size_t s = 0; s < nmo_; ++s){
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct   = two_electron_integrals[INDEX4(p,r,q,s)];
                        double exchange = two_electron_integrals[INDEX4(p,s,q,r)];
                        size_t index = aptei_index(p,q,r,s);
                        aphys_tei_aa[index] = direct - exchange;
                        aphys_tei_ab[index] = direct;
                        aphys_tei_bb[index] = direct - exchange;
                    }
                }
            }
        }

        // Deallocate temp memory
        delete[] two_electron_integrals;
    }else{
        double* two_electron_integrals = new double[num_tei];
        // Alpha-alpha integrals
        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries
        for (size_t pqrs = 0; pqrs < num_tei; ++pqrs) two_electron_integrals[pqrs] = 0.0;

        // Read the integrals
        struct iwlbuf V_AAAA;
        iwl_buf_init(&V_AAAA,PSIF_MO_AA_TEI, 0.0, 1, 1);
        iwl_buf_rd_all(&V_AAAA, two_electron_integrals, myioff, myioff, 0, myioff, 0, outfile);
        iwl_buf_close(&V_AAAA, 1);

        for (size_t p = 0; p < nmo_; ++p){
            for (size_t q = 0; q < nmo_; ++q){
                for (size_t r = 0; r < nmo_; ++r){
                    for (size_t s = 0; s < nmo_; ++s){
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct   = two_electron_integrals[INDEX4(p,r,q,s)];
                        double exchange = two_electron_integrals[INDEX4(p,s,q,r)];
                        size_t index = aptei_index(p,q,r,s);
                        aphys_tei_aa[index] = direct - exchange;
                    }
                }
            }
        }

        // Beta-beta integrals
        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries
        for (size_t pqrs = 0; pqrs < num_tei; ++pqrs) two_electron_integrals[pqrs] = 0.0;

        // Read the integrals
        struct iwlbuf V_BBBB;
        iwl_buf_init(&V_BBBB,PSIF_MO_BB_TEI, 0.0, 1, 1);
        iwl_buf_rd_all(&V_BBBB, two_electron_integrals, myioff, myioff, 0, myioff, 0, outfile);
        iwl_buf_close(&V_BBBB, 1);

        for (size_t p = 0; p < nmo_; ++p){
            for (size_t q = 0; q < nmo_; ++q){
                for (size_t r = 0; r < nmo_; ++r){
                    for (size_t s = 0; s < nmo_; ++s){
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct   = two_electron_integrals[INDEX4(p,r,q,s)];
                        double exchange = two_electron_integrals[INDEX4(p,s,q,r)];
                        size_t index = aptei_index(p,q,r,s);
                        aphys_tei_bb[index] = direct - exchange;
                    }
                }
            }
        }
        // Deallocate temp memory
        delete[] two_electron_integrals;

        // Alpha-beta integrals
        Matrix Tei(num_oei,num_oei);
        double** two_electron_integrals_ab = Tei.pointer();
        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries
        for (size_t pq = 0; pq < num_oei; ++pq){
            for (size_t rs = 0; rs < num_oei; ++rs){
                two_electron_integrals_ab[pq][rs] = 0.0;
            }
        }

        // Read the integrals
        struct iwlbuf V_AABB;
        iwl_buf_init(&V_AABB,PSIF_MO_AB_TEI, 0.0, 1, 1);
        iwl_buf_rd_all2(&V_AABB, two_electron_integrals_ab, myioff, myioff, 0, myioff, 0, outfile);
        iwl_buf_close(&V_AABB, 1);

        for (size_t p = 0; p < nmo_; ++p){
            for (size_t q = 0; q < nmo_; ++q){
                for (size_t r = 0; r < nmo_; ++r){
                    for (size_t s = 0; s < nmo_; ++s){
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct = two_electron_integrals_ab[INDEX2(p,r)][INDEX2(q,s)];
                        size_t index = aptei_index(p,q,r,s);
                        aphys_tei_ab[index] = direct;
                    }
                }
            }
        }
    }
    delete[] myioff;
}

void ExplorerIntegrals::make_diagonal_integrals()
{
//    for (int p = 0; p < nmo_; ++p){
//        diagonal_one_electron_integrals_a[p] = oei_a(p,p);
//        diagonal_one_electron_integrals_b[p] = oei_b(p,p);
//    }

    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

void ExplorerIntegrals::make_fock_matrix(bool* Ia, bool* Ib)
{
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            // Builf Fock Diagonal alpha-alpha
            fock_matrix_a[p * nmo_ + q] = oei_a(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < nmo_; ++k) {
                if (Ia[k]) {
                    //                    fock_matrix_a[p * nmo_ + q] += rtei(p,q,k,k) - rtei(p,k,q,k);
                    fock_matrix_a[p * nmo_ + q] += aptei_aa(p,k,q,k);// -  rtei(p,q,k,k) - rtei(p,k,q,k);
                }
                if (Ib[k]) {
                    //                    fock_matrix_a[p * nmo_ + q] += rtei(p,q,k,k);
                    fock_matrix_a[p * nmo_ + q] += aptei_ab(p,k,q,k);
                }
            }
            fock_matrix_b[p * nmo_ + q] = oei_b(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < nmo_; ++k) {
                if (Ib[k]) {
                    //                    fock_matrix_b[p * nmo_ + q] += rtei(p,q,k,k) - rtei(p,k,q,k);
                    fock_matrix_b[p * nmo_ + q] += aptei_bb(p,k,q,k);
                }
                if (Ia[k]) {
                    fock_matrix_b[p * nmo_ + q] += aptei_ab(p,k,q,k);//rtei(p,q,k,k);
                }
            }
        }
    }
}

void ExplorerIntegrals::make_fock_diagonal(bool* Ia, bool* Ib, std::pair<std::vector<double>, std::vector<double> > &fock_diagonals)
{
    std::vector<double>& fock_diagonal_alpha = fock_diagonals.first;
    std::vector<double>& fock_diagonal_beta = fock_diagonals.second;
    for(size_t p = 0; p < nmo_; ++p){
        // Builf Fock Diagonal alpha-alpha
        fock_diagonal_alpha[p] =  oei_a(p,p);// roei(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < nmo_; ++k) {
            if (Ia[k]) {
                //                fock_diagonal_alpha[p] += diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
                fock_diagonal_alpha[p] += diag_aptei_aa(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ib[k]) {
                //                fock_diagonal_alpha[p] += diag_c_rtei(p,k); //rtei(p,p,k,k);
                fock_diagonal_alpha[p] += diag_aptei_ab(p,k); //rtei(p,p,k,k);
            }
        }
        fock_diagonal_beta[p] =  oei_b(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < nmo_; ++k) {
            if (Ib[k]) {
                //                fock_diagonal_beta[p] += diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
                fock_diagonal_beta[p] += diag_aptei_bb(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ia[k]) {
                //                fock_diagonal_beta[p] += diag_c_rtei(p,k); //rtei(p,p,k,k);
                fock_diagonal_beta[p] += diag_aptei_ab(p,k); //rtei(p,p,k,k);
            }
        }
    }
}

void ExplorerIntegrals::make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double> &fock_diagonal)
{
    for(size_t p = 0; p < nmo_; ++p){
        // Builf Fock Diagonal alpha-alpha
        fock_diagonal[p] = oei_a(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < nmo_; ++k) {
            if (Ia[k]) {
                fock_diagonal[p] += diag_aptei_aa(p,k);  //diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ib[k]) {
                fock_diagonal[p] += diag_aptei_ab(p,k); // diag_c_rtei(p,k); //rtei(p,p,k,k);
            }
        }
    }
}

void ExplorerIntegrals::make_beta_fock_diagonal(bool* Ia, bool* Ib, std::vector<double> &fock_diagonals)
{
    for(size_t p = 0; p < nmo_; ++p){
        fock_diagonals[p] = oei_b(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < nmo_; ++k) {
            if (Ia[k]) {
                fock_diagonals[p] += diag_aptei_ab(p,k);  //diag_c_rtei(p,k); //rtei(p,p,k,k);
            }
            if (Ib[k]) {
                fock_diagonals[p] += diag_aptei_bb(p,k);  //diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
        }
    }
}


void ExplorerIntegrals::freeze_core_orbitals()
{
    compute_frozen_core_energy();
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs){
        resort_integrals_after_freezing();
    }
}

void ExplorerIntegrals::compute_frozen_core_energy()
{
    core_energy_ = 0.0;

    for (int hi = 0, p = 0; hi < nirrep_; ++hi){
        for (int i = 0; i < frzcpi_[hi]; ++i){
            core_energy_ += oei_a(p + i,p + i) + oei_b(p + i,p + i);
            for (int hj = 0, q = 0; hj < nirrep_; ++hj){
                for (int j = 0; j < frzcpi_[hj]; ++j){
                    //                    core_energy_ += diag_ce_rtei(p + i,q + i) + diag_c_rtei(p + i,q + i);
                    core_energy_ += 0.5 * diag_aptei_aa(p + i,q + i) + 0.5 * diag_aptei_bb(p + i,q + i)  + diag_aptei_ab(p + i,q + i);
                }
                q += nmopi_[hj]; // orbital offset for the irrep hj
            }
        }
        p += nmopi_[hi]; // orbital offset for the irrep hi
    }

    fprintf(outfile,"\n  Frozen-core energy = %20.12f a.u.",core_energy_);
}

void ExplorerIntegrals::compute_frozen_one_body_operator()
{
    fprintf(outfile,"\n  Creating a modified one-body operator.");

    // Modify the active part of H to include the core effects;
    size_t f = 0;
    for (int hi = 0; hi < nirrep_; ++hi){
        for (int i = 0; i < frzcpi_[hi]; ++i){
            size_t r = f + i;
            fprintf(outfile,"\n  Freezing MO %zu",r);
            for(size_t p = 0; p < nmo_; ++p){
                for(size_t q = 0; q < nmo_; ++q){
                    one_electron_integrals_a[p * nmo_ + q] += aptei_aa(r,p,r,q) + aptei_ab(r,p,r,q);
                    one_electron_integrals_b[p * nmo_ + q] += aptei_bb(r,p,r,q) + aptei_ab(r,p,r,q);
                }
            }
        }
        f += nmopi_[hi];
    }
}

void ExplorerIntegrals::resort_integrals_after_freezing()
{
    fprintf(outfile,"\n  Resorting integrals after freezing core.");

    // Create a mapping array cmo2mo that tell me (cmo2mo[i]) where to find the i-th cmo.
    std::vector<size_t> cmo2mo;
    for (int h = 0, p = 0, q = 0; h < nirrep_; ++h){
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r){
            cmo2mo[p] = q;
            p++;
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }

    // Resort the integrals
    resort_two(one_electron_integrals_a,cmo2mo);
    resort_two(one_electron_integrals_b,cmo2mo);
    resort_two(diagonal_aphys_tei_aa,cmo2mo);
    resort_two(diagonal_aphys_tei_ab,cmo2mo);
    resort_two(diagonal_aphys_tei_bb,cmo2mo);
    resort_four(aphys_tei_aa,cmo2mo);
    resort_four(aphys_tei_ab,cmo2mo);
    resort_four(aphys_tei_bb,cmo2mo);
}


void ExplorerIntegrals::resort_two(double*& ints,std::vector<size_t>& map)
{
    // Store the integrals in a temporary array of dimension nmo x nmo
    double* temp_ints = new double[nmo_ * nmo_];
    for (size_t p = 0; p < nmo_ * nmo_; ++p){
        temp_ints[p] = 0.0;
    }
    for (size_t p = 0; p < ncmo_; ++p){
        for (size_t q = 0; q < ncmo_; ++q){
            temp_ints[p * ncmo_ + q] = ints[map[p] * nmo_ + map[q]];
        }
    }
    // Delete old integrals and assign the pointer
    delete[] ints;
    ints = temp_ints;
}

void ExplorerIntegrals::resort_four(double*& ints,std::vector<size_t>& map)
{
    // Store the integrals in a temporary array
    double* temp_ints = new double[num_aptei];
    for (size_t p = 0; p < num_aptei; ++p){
        temp_ints[p] = 0.0;
    }
    for (size_t p = 0; p < ncmo_; ++p){
        for (size_t q = 0; q < ncmo_; ++q){
            for (size_t r = 0; r < ncmo_; ++r){
                for (size_t s = 0; s < ncmo_; ++s){
                    size_t pqrs_cmo = ncmo3_ * p + ncmo2_ * q + ncmo_ * r + s;
                    size_t pqrs_mo  = nmo_ * nmo_ * nmo_ * map[p] + nmo_ * nmo_ * map[q] + nmo_ * map[r] + map[s];
                    temp_ints[pqrs_cmo] = ints[pqrs_mo];
                }
            }
        }
    }
    // Delete old integrals and assign the pointer
    delete[] ints;
    ints = temp_ints;
}

void ExplorerIntegrals::set_oei(double** ints,bool alpha)
{
    double* p_oei = alpha ? one_electron_integrals_a : one_electron_integrals_b;
    for (int p = 0; p < nmo_; ++p){
        for (int q = 0; q < nmo_; ++q){
            p_oei[p * nmo_ + q] = ints[p][q];
        }
    }
}

void ExplorerIntegrals::set_oei(size_t p, size_t q,double value,bool alpha)
{
    double* p_oei = alpha ? one_electron_integrals_a : one_electron_integrals_b;
    p_oei[p * nmo_ + q] = value;
}

/// This functions receives integrals stored in the format
/// ints[p][q][r][s] = v_{pq}^{rs}
void ExplorerIntegrals::set_tei(double**** ints,bool alpha1,bool alpha2)
{
    double* p_tei;
    if (alpha1 == true and alpha2 == true) p_tei = aphys_tei_aa;
    if (alpha1 == true and alpha2 == false) p_tei = aphys_tei_ab;
    if (alpha1 == false and alpha2 == false) p_tei = aphys_tei_bb;
    for (size_t p = 0; p < nmo_; ++p){
        for (size_t q = 0; q < nmo_; ++q){
            for (size_t r = 0; r < nmo_; ++r){
                for (size_t s = 0; s < nmo_; ++s){
                    size_t index = aptei_index(p,q,r,s);
                    double integral = ints[p][q][r][s];
                    if (std::fabs(integral) > 1.0e-9)
                        fprintf(outfile,"\n (%zu %zu | %zu %zu) = v_{%zu %zu}^{%zu %zu} = [%zu] = %f",p,r,q,s,p,q,r,s,index,integral);
                    p_tei[index] = integral;
                }
            }
        }
    }
}

void ExplorerIntegrals::set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2)
{
    double* p_tei;
    if (alpha1 == true and alpha2 == true) p_tei = aphys_tei_aa;
    if (alpha1 == true and alpha2 == false) p_tei = aphys_tei_ab;
    if (alpha1 == false and alpha2 == false) p_tei = aphys_tei_bb;
    size_t index = aptei_index(p,q,r,s);
    p_tei[index] = value;
    //    if (std::fabs(value) > 1.0e-9)
    //        fprintf(outfile,"\n (%zu %zu | %zu %zu) = v_{%zu %zu}^{%zu %zu} = [%zu] = %f",p,r,q,s,p,q,r,s,index,value);
}

///**
// * Make the one electron intermediates
// * h'_pq = h_pq - 1/2 \sum_r (pr|qr)
// */
//void Integrals::make_h()
//{
//  for(size_t p = 0; p < nmo; ++p){
//    for(size_t q = 0; q < nmo; ++q){
//      h_aa[p][q] = oei_aa[p][q];
//      h_bb[p][q] = oei_bb[p][q];
//      for(size_t r = 0; r < nmo; ++r){
//        h_aa[p][q] -= 0.5 * aaaa(p,r,q,r);
//        h_bb[p][q] -= 0.5 * bbbb(p,r,q,r);
//      }
//    }
//  }
//}

///**
// * Make the fock matrix
// */
//void Integrals::make_f()
//{
//  int nrefs = moinfo->get_nrefs();
//  int nall = moinfo->get_nall();
//  for(int mu = 0 ; mu < nrefs; ++mu){
//    std::vector<int> occupation = moinfo->get_determinant(mu);
//    for(size_t p = 0; p < nmo; ++p){
//      for(size_t q = 0; q < nmo; ++q){
//        // Builf Fock Diagonal alpha-alpha
//        f_aa[mu][p][q] = oei_aa[p][q];
//        // Add the non-frozen alfa part, the forzen core part is already included in oei
//        for (int k = 0; k < moinfo->get_nall(); ++k) {
//          size_t k_f = moinfo->get_all_to_mo()[k];
//          if (occupation[k]) {
//            f_aa[mu][p][q] += aaaa(p,q,k_f,k_f) - aaaa(p,k_f,q,k_f);
//          }
//          if (occupation[k + nall]) {
//            f_aa[mu][p][q] += aabb(p,q,k_f,k_f);
//          }
//        }

//        // Builf Fock Diagonal beta-beta
//        f_bb[mu][p][q] = oei_bb[p][q];
//        // Add the non-frozen beta part, the forzen core part is already included in oei
//        for (int k = 0; k < moinfo->get_nall(); ++k) {
//          size_t k_f = moinfo->get_all_to_mo()[k];
//          if (occupation[k]) {
//            f_bb[mu][p][q] += bbaa(p,q,k_f,k_f);
//          }
//          if (occupation[k + nall]) {
//            f_bb[mu][p][q] += bbbb(p,q,k_f,k_f) - bbbb(p,k_f,q,k_f);
//          }
//        }
//      }
//    }
//    if(options_.get_int("DEBUG") > 6){
//      fprintf(outfile,"\n  alfa-alfa fock matrix for reference %d",mu);
//      mat_print(f_aa[mu],nmo,nmo,outfile);
//      fprintf(outfile,"\n  beta-beta fock matrix for reference %d",mu);
//      mat_print(f_bb[mu],nmo,nmo,outfile);
//    }
//  }
//}

///**
// * Make the average fock matrix
// */
//void Integrals::make_f_avg(double** opdm_aa,double** opdm_bb)
//{
//  int nall = moinfo->get_nall();
//  for(size_t p = 0; p < nmo; ++p){
//    for(size_t q = 0; q < nmo; ++q){
//      // Builf Fock Diagonal alpha-alpha
//      f_avg_aa[p][q] = oei_aa[p][q];
//      // Add the non-frozen alfa part, the forzen core part is already included in oei
//      for (int r = 0; r < nall; ++r) {
//        for (int s = 0; s < nall; ++s) {
//          size_t r_f = moinfo->get_all_to_mo()[r];
//          size_t s_f = moinfo->get_all_to_mo()[s];
//          f_avg_aa[p][q] += opdm_aa[r][s] * (aaaa(p,q,r_f,s_f) - aaaa(p,r_f,q,s_f));
//          f_avg_aa[p][q] += opdm_bb[r][s] *  aabb(p,q,r_f,s_f);
//        }
//      }

//      // Builf Fock Diagonal beta-beta
//      f_avg_bb[p][q] = oei_bb[p][q];
//      // Add the non-frozen beta part, the forzen core part is already included in oei
//      // Add the non-frozen alfa part, the forzen core part is already included in oei
//      for (int r = 0; r < nall; ++r) {
//        for (int s = 0; s < nall; ++s) {
//          size_t r_f = moinfo->get_all_to_mo()[r];
//          size_t s_f = moinfo->get_all_to_mo()[s];
//          f_avg_bb[p][q] += opdm_aa[r][s] *  bbaa(p,q,r_f,s_f);
//          f_avg_bb[p][q] += opdm_bb[r][s] * (bbbb(p,q,r_f,s_f) - bbbb(p,r_f,q,s_f));
//        }
//      }
//    }
//  }
//  if(options_.get_int("DEBUG") > 6){
//    fprintf(outfile,"\n  alfa-alfa average fock matrix");
//    mat_print(f_avg_aa,nmo,nmo,outfile);
//    fprintf(outfile,"\n  beta-beta average fock matrix");
//    mat_print(f_avg_bb,nmo,nmo,outfile);
//  }
//}



///**
// * Compute the denominator corresponding to the alfa excitation
// * (i_1,...,i_k,a_1,...,a_k) =  a+(a_k) ... a+(a_2) a+(a_1) a-(i_k) ... a-(i_2) a-(i_1)
// * @param occ the list of occupied labels numbered according to the all space
// * @param vir the list of virtual labels numbered according to the all space
// * @param n the number of labels
// * @param mu the reference determinant
// */
//double Integrals::get_alfa_denominator(const int* occvir,int n,int mu)
//{
//  double value = 0.0;
//  for (int i = 0; i < n; ++i) {
//    int i_f = all_to_mo[occvir[i]];
//    int a_f = all_to_mo[occvir[i + n]];
//    value += f_aa[mu][i_f][i_f];
//    value -= f_aa[mu][a_f][a_f];
//  }
//  return value;
//}

///**
// * Compute the denominator corresponding to the beta excitation
// * (i_1,...,i_k,a_1,...,a_k) =  a+(a_k) ... a+(a_2) a+(a_1) a-(i_k) ... a-(i_2) a-(i_1)
// * @param occ the list of occupied labels numbered according to the all space
// * @param vir the list of virtual labels numbered according to the all space
// * @param n the number of labels
// * @param mu the reference determinant
// */
//double Integrals::get_beta_denominator(const int* occvir,int n,int mu)
//{
//  double value = 0.0;
//  for (int i = 0; i < n; ++i) {
//    int i_f = all_to_mo[occvir[i]];
//    int a_f = all_to_mo[occvir[i + n]];
//    value += f_bb[mu][i_f][i_f];
//    value -= f_bb[mu][a_f][a_f];
//  }
//  return value;
//}

///**
// * Compute the denominator corresponding to the alfa excitation
// * (i_1,...,i_k,a_1,...,a_k) =  a+(a_k) ... a+(a_2) a+(a_1) a-(i_k) ... a-(i_2) a-(i_1)
// * @param occ the list of occupied labels numbered according to the all space
// * @param vir the list of virtual labels numbered according to the all space
// * @param n the number of labels
// * @param mu the reference determinant
// */
//double Integrals::get_alfa_denominator(int* occ,int* vir,int n,int mu)
//{
//  double value = 0.0;
//  for (int i = 0; i < n; ++i) {
//    int i_f = all_to_mo[occ[i]];
//    int a_f = all_to_mo[vir[i]];
//    value += f_aa[mu][i_f][i_f];
//    value -= f_aa[mu][a_f][a_f];
//  }
//  return value;
//}

///**
// * Compute the denominator corresponding to the beta excitation
// * (i_1,...,i_k,a_1,...,a_k) =  a+(a_k) ... a+(a_2) a+(a_1) a-(i_k) ... a-(i_2) a-(i_1)
// * @param occ the list of occupied labels numbered according to the all space
// * @param vir the list of virtual labels numbered according to the all space
// * @param n the number of labels
// * @param mu the reference determinant
// */
//double Integrals::get_beta_denominator(int* occ,int* vir,int n,int mu)
//{
//  double value = 0.0;
//  for (int i = 0; i < n; ++i) {
//    int i_f = all_to_mo[occ[i]];
//    int a_f = all_to_mo[vir[i]];
//    value += f_bb[mu][i_f][i_f];
//    value -= f_bb[mu][a_f][a_f];
//  }
//  return value;
//}

///**
// * Compute the denominator corresponding to the alfa excitation using the average fock matrix
// * (i_1,...,i_k,a_1,...,a_k) =  a+(a_k) ... a+(a_2) a+(a_1) a-(i_k) ... a-(i_2) a-(i_1)
// * @param occ the list of occupied labels numbered according to the all space
// * @param vir the list of virtual labels numbered according to the all space
// * @param n the number of labels
// */
//double Integrals::get_alfa_avg_denominator(const int* occvir,int n)
//{
//  double value = 0.0;
//  for (int i = 0; i < n; ++i) {
//    int i_f = all_to_mo[occvir[i]];
//    int a_f = all_to_mo[occvir[i + n]];
//    value += f_avg_aa[i_f][i_f];
//    value -= f_avg_aa[a_f][a_f];
//  }
//  return value;
//}

///**
// * Compute the denominator corresponding to the beta excitation using the average fock matrix
// * (i_1,...,i_k,a_1,...,a_k) =  a+(a_k) ... a+(a_2) a+(a_1) a-(i_k) ... a-(i_2) a-(i_1)
// * @param occ the list of occupied labels numbered according to the all space
// * @param vir the list of virtual labels numbered according to the all space
// * @param n the number of labels
// * @param mu the reference determinant
// */
//double Integrals::get_beta_avg_denominator(const int* occvir,int n)
//{
//  double value = 0.0;
//  for (int i = 0; i < n; ++i) {
//    int i_f = all_to_mo[occvir[i]];
//    int a_f = all_to_mo[occvir[i + n]];
//    value += f_avg_bb[i_f][i_f];
//    value -= f_avg_bb[a_f][a_f];
//  }
//  return value;
//}

}} // End Namespaces



//    // Create the mapping from the pairs [A,A] to the irrep and the index
//    nmo_ = wfn->nmo();
//    pair_irrep_map.resize(nmo_ * nmo_);
//    pair_index_map.resize(nmo_ * nmo_);
//    Dimension nmopi = wfn->nmopi();
//    vector<int> mooff;
//    vector<int> irrep_off(nirrep_,0);
//    mooff.push_back(0);
//    for (int h = 1; h < nirrep_; ++h) mooff.push_back(mooff[h-1] + nmopi[h]);
//    for (int hp = 0; hp < nirrep_; ++hp){
//        for (int hq = 0; hq < nirrep_; ++hq){
//            for (int p = 0; p < nmopi[hp]; ++p){
//                for (int q = 0; q < nmopi[hq]; ++q){
//                    int absp = p + mooff[hp];
//                    int absq = q + mooff[hq];
//                    int pq_irrep = hp ^ hq;
//                    int pq_index = absp * nmo_ + absq;
//                    pair_irrep_map[pq_index] = pq_irrep;
//                    pair_index_map[pq_index] = irrep_off[pq_irrep];
//                    irrep_off[pq_irrep] += 1;
//                }
//            }
//        }
//    }



//    ints_ = new IntegralTransform(wfn, spaces, IntegralTransform::Restricted);
//    // Use the IntegralTransform object's DPD instance, for convenience
//    dpd_set_default(ints_->get_dpd_id());

//    dpdbuf4 K;
//    psio->open(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
//    // To only process the permutationally unique integrals, change the ID("[A,A]") to ID("[A>=A]+")
//    global_dpd_->buf4_init(&K, PSIF_LIBTRANS_DPD, 0, ID("[A,A]"), ID("[A,A]"),
//                  ID("[A>=A]+"), ID("[A>=A]+"), 0, "MO Ints (AA|AA)");
//    for(int h = 0; h < nirrep_; ++h){
//        global_dpd_->buf4_mat_irrep_init(&K, h);
//        global_dpd_->buf4_mat_irrep_rd(&K, h);
//        for(int pq = 0; pq < K.params->rowtot[h]; ++pq){
//            size_t p = K.params->roworb[h][pq][0];
//            size_t q = K.params->roworb[h][pq][1];
//            int psym = K.params->psym[p];
//            int qsym = K.params->qsym[q];
//            int prel = p - K.params->poff[psym];
//            int qrel = q - K.params->qoff[qsym];
//            for(int rs = 0; rs < K.params->coltot[h]; ++rs){
//                size_t r = K.params->colorb[h][rs][0];
//                size_t s = K.params->colorb[h][rs][1];
//                int rsym = K.params->rsym[r];
//                int ssym = K.params->ssym[s];
//                int rrel = r - K.params->roff[rsym];
//                int srel = s - K.params->soff[ssym];
//                // Print out the absolute orbital numbers, the relative (within irrep)
//                // numbers, the symmetries, and the integral itself
////                fprintf(outfile, "(%2d %2d | %2d %2d) = %16.10f, "
////                                 "symmetries = (%1d %1d | %1d %1d), "
////                                 "relative indices = (%2d %2d | %2d %2d)\n",
////                                 p, q, r, s, K.matrix[h][pq][rs],
////                                 psym, qsym, rsym, ssym,
////                                 prel, qrel, rrel, srel);
//                two_electron_integrals[INDEX4(p,q,r,s)] = K.matrix[h][pq][rs];
//            }
//        }
//        global_dpd_->buf4_mat_irrep_close(&K, h);
//    }
//    global_dpd_->buf4_close(&K);
//    psio->close(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);



//  // Assume all the integrals are stored regardless of symmetry
//  nmo = wfn->nmo();
//  ntei = four(nmo-1,nmo-1,nmo-1,nmo-1) + 1;
//  noei = ioff[nmo];

//  allocate1(int,all_to_mo,moinfo->get_nall());
//  for(int p = 0; p < moinfo->get_nall(); ++p)
//    all_to_mo[p] = moinfo->get_all_to_mo()[p];

//  allocate2(double,oei_aa,nmo,nmo);
//  allocate2(double,oei_bb,nmo,nmo);
//  allocate2(double,h_aa,nmo,nmo);
//  allocate2(double,h_bb,nmo,nmo);
//  allocate1(double,tei_aaaa,ntei);
//  allocate1(double,tei_bbbb,ntei);
//  allocate2(double,tei_aabb,noei,noei);

//  // Fock matrix
//  int nrefs = moinfo->get_nrefs();
//  allocate1(double**,f_aa,nrefs);
//  allocate1(double**,f_bb,nrefs);
//  for (int mu = 0; mu < nrefs; ++mu) {
//    allocate2(double,f_aa[mu],nmo,nmo);
//    allocate2(double,f_bb[mu],nmo,nmo);
//  }
//  allocate2(double,f_avg_aa,nmo,nmo);
//  allocate2(double,f_avg_bb,nmo,nmo);
//  // N.B. assumes all arrays are zeroed

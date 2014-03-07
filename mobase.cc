#include <cmath>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include "multidimensional_arrays.h"

#include "mobase.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

MOBase::MOBase(Options &options, ExplorerIntegrals* ints, TwoIndex G1aa, TwoIndex G1bb)
    : ints_(ints), options_(options)
{
    startup(G1aa,G1bb);
}

MOBase::~MOBase()
{
    release();
}

void MOBase::startup(TwoIndex G1aa,TwoIndex G1bb)
{
    // Extract data from the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    nirrep_ = wfn->nirrep();
    nmo_ = wfn->nmo();
    nmopi_ = wfn->nmopi();

    print_ = options_.get_int("PRINT");

    //    frzcpi_ = Dimension(nirrep_);
    //    frzvpi_ = Dimension(nirrep_);

    allocate();
    sort_integrals();

    loop_mo_p{
        No_.a[p] = G1aa[p][p];
        No_.b[p] = G1bb[p][p];
        Nv_.a[p] = 1.0 - G1aa[p][p];
        Nv_.b[p] = 1.0 - G1bb[p][p];
    }
    loop_mo_p loop_mo_q{
        G1_.aa[p][q] = G1aa[p][q];
        G1_.bb[p][q] = G1bb[p][q];
        E1_.aa[p][q] = (p == q ? 1.0 : 0.0) - G1_.aa[p][q];
        E1_.bb[p][q] = (p == q ? 1.0 : 0.0) - G1_.bb[p][q];
    }

    build_fock();
}

void MOBase::allocate()
{
    No_.a = new double[nmo_];
    No_.b = new double[nmo_];
    Nv_.a = new double[nmo_];
    Nv_.b = new double[nmo_];
    allocate(G1_);
    allocate(E1_);
    allocate(G2_);
    allocate(L2_);
    allocate(H1_);
    allocate(F_);
    allocate(V_);
}

void MOBase::release()
{
    delete[] No_.a;
    delete[] No_.b;
    delete[] Nv_.a;
    delete[] Nv_.b;
    release(G1_);
    release(E1_);
    release(G2_);
    release(L2_);
    release(H1_);
    release(F_);
    release(V_);
}

void MOBase::allocate(MOTwoIndex&  two_index)
{
    init_matrix<double>(two_index.aa,nmo_,nmo_);
    init_matrix<double>(two_index.bb,nmo_,nmo_);
}

void MOBase::allocate(MOFourIndex& four_index)
{
    init_matrix<double>(four_index.aaaa,nmo_,nmo_,nmo_,nmo_);
    init_matrix<double>(four_index.abab,nmo_,nmo_,nmo_,nmo_);
    init_matrix<double>(four_index.bbbb,nmo_,nmo_,nmo_,nmo_);
}

void MOBase::release(MOTwoIndex&  two_index)
{
    free_matrix<double>(two_index.aa,nmo_,nmo_);
    free_matrix<double>(two_index.bb,nmo_,nmo_);
}

void MOBase::release(MOFourIndex& four_index)
{
    free_matrix<double>(four_index.aaaa,nmo_,nmo_,nmo_,nmo_);
    free_matrix<double>(four_index.abab,nmo_,nmo_,nmo_,nmo_);
    free_matrix<double>(four_index.bbbb,nmo_,nmo_,nmo_,nmo_);
}

void MOBase::sort_integrals()
{
    loop_mo_p loop_mo_q{
        H1_.aa[p][q] = ints_->oei_a(p,q);
        H1_.bb[p][q] = ints_->oei_b(p,q);
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        V_.aaaa[p][q][r][s] = ints_->aptei_aa(p,q,r,s); //ints_->rtei(p,r,q,s) - ints_->rtei(p,s,q,r);
        V_.abab[p][q][r][s] = ints_->aptei_ab(p,q,r,s); //ints_->rtei(p,r,q,s);
        V_.bbbb[p][q][r][s] = ints_->aptei_bb(p,q,r,s); //ints_->rtei(p,r,q,s) - ints_->rtei(p,s,q,r);
    }
}

void MOBase::build_fock()
{
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    boost::shared_ptr<Molecule> molecule_ = wfn->molecule();
    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();

    // Compute the reference energy
    E0_ = nuclear_repulsion_energy_;
    loop_mo_p loop_mo_q{
        E0_ += H1_.aa[p][q] * G1_.aa[q][p];
        E0_ += H1_.bb[p][q] * G1_.bb[q][p];
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        E0_ += 0.25 * V_.aaaa[p][q][r][s] * (G1_.aa[p][r] * G1_.aa[q][s] - G1_.aa[p][s] * G1_.aa[q][r]);
        E0_ += 0.25 * V_.bbbb[p][q][r][s] * (G1_.bb[p][r] * G1_.bb[q][s] - G1_.bb[p][s] * G1_.bb[q][r]);
        E0_ +=  1.0 * V_.abab[p][q][r][s] * G1_.aa[p][r] * G1_.bb[q][s];
    }
    // Compute the fock matrix
    loop_mo_p loop_mo_q{
        F_.aa[p][q] = H1_.aa[p][q];
        F_.bb[p][q] = H1_.bb[p][q];
        loop_mo_r loop_mo_s{
            F_.aa[p][q] += V_.aaaa[p][r][q][s] * G1_.aa[s][r] + V_.abab[p][r][q][s] * G1_.bb[s][r];
            F_.bb[p][q] += V_.bbbb[p][r][q][s] * G1_.bb[s][r] + V_.abab[r][p][s][q] * G1_.aa[s][r];
        }
    }

    fprintf(outfile,"\n  The energy of the reference is: %20.12f Eh",E0_);
    fprintf(outfile,"\n  Diagonal elements of the Fock matrix:");
    fprintf(outfile,"\n  SO            Epsilon         ON");
    loop_mo_p {
        fprintf(outfile,"\n  %2d  %20.12f   %8.6f  %20.12f   %8.6f",p,F_.aa[p][p],G1_.aa[p][p],F_.bb[p][p],G1_.bb[p][p]);
    }
}

void MOBase::add(double fA,MOTwoIndex& A, double fB, MOTwoIndex& B)
{
    loop_mo_p loop_mo_q{
        B.aa[p][q] = fA * A.aa[p][q] + fB * B.aa[p][q];
    }
    loop_mo_p loop_mo_q{
        B.bb[p][q] = fA * A.bb[p][q] + fB * B.bb[p][q];
    }
}

void MOBase::add(double fA,MOFourIndex& A, double fB, MOFourIndex& B)
{
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        B.aaaa[p][q][r][s] = fA * A.aaaa[p][q][r][s] + fB * B.aaaa[p][q][r][s];
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        B.abab[p][q][r][s] = fA * A.abab[p][q][r][s] + fB * B.abab[p][q][r][s];
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        B.bbbb[p][q][r][s] = fA * A.bbbb[p][q][r][s] + fB * B.bbbb[p][q][r][s];
    }
}

double MOBase::norm(MOTwoIndex& A)
{
    double norm = 0.0;
    loop_mo_p loop_mo_q{
        norm += std::pow(A.aa[p][q],2.0);
        norm += std::pow(A.bb[p][q],2.0);
    }
    return std::sqrt(norm);
}

double MOBase::norm(MOFourIndex& A)
{
    double norm = 0.0;
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        norm += std::pow(A.aaaa[p][q][r][s],2.0);
        norm += std::pow(A.abab[p][q][r][s],2.0);
        norm += std::pow(A.bbbb[p][q][r][s],2.0);
    }
    return std::sqrt(0.25 * norm);
}

void MOBase::zero(MOTwoIndex& A)
{
    loop_mo_p loop_mo_q{
        A.aa[p][q] = 0.0;
        A.bb[p][q] = 0.0;
    }
}

void MOBase::zero(MOFourIndex& A)
{
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        A.aaaa[p][q][r][s] = 0.0;
        A.abab[p][q][r][s] = 0.0;
        A.bbbb[p][q][r][s] = 0.0;
    }
}

}} // EndNamespaces

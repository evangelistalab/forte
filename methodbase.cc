#include "methodbase.h"

#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

namespace psi{ namespace libadaptive{

MethodBase::MethodBase(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), ints_(ints)
{
    // Copy the wavefunction information
    copy(wfn);
    startup();
}

MethodBase::~MethodBase()
{
    cleanup();
}

void MethodBase::startup()
{
}

void MethodBase::cleanup()
{
}
    
    //void MethodBase::sort_integrals()
    //{
    ////    loop_mo_p loop_mo_q{
    ////        H1_.aa[p][q] = ints_->oei_a(p,q);
    ////        H1_.bb[p][q] = ints_->oei_b(p,q);
    ////    }
    ////    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
    ////        V_.aaaa[p][q][r][s] = ints_->aptei_aa(p,q,r,s); //ints_->rtei(p,r,q,s) - ints_->rtei(p,s,q,r);
    ////        V_.abab[p][q][r][s] = ints_->aptei_ab(p,q,r,s); //ints_->rtei(p,r,q,s);
    ////        V_.bbbb[p][q][r][s] = ints_->aptei_bb(p,q,r,s); //ints_->rtei(p,r,q,s) - ints_->rtei(p,s,q,r);
    ////    }
    //}
    
    //void MethodBase::build_fock()
    //{
    ////    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    
    ////    boost::shared_ptr<Molecule> molecule_ = wfn->molecule();
    ////    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();
    
    ////    // Compute the reference energy
    ////    E0_ = nuclear_repulsion_energy_;
    ////    loop_mo_p loop_mo_q{
    ////        E0_ += H1_.aa[p][q] * G1_.aa[q][p];
    ////        E0_ += H1_.bb[p][q] * G1_.bb[q][p];
    ////    }
    ////    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
    ////        E0_ += 0.25 * V_.aaaa[p][q][r][s] * (G1_.aa[p][r] * G1_.aa[q][s] - G1_.aa[p][s] * G1_.aa[q][r]);
    ////        E0_ += 0.25 * V_.bbbb[p][q][r][s] * (G1_.bb[p][r] * G1_.bb[q][s] - G1_.bb[p][s] * G1_.bb[q][r]);
    ////        E0_ +=  1.0 * V_.abab[p][q][r][s] * G1_.aa[p][r] * G1_.bb[q][s];
    ////    }
    ////    // Compute the fock matrix
    ////    loop_mo_p loop_mo_q{
    ////        F_.aa[p][q] = H1_.aa[p][q];
    ////        F_.bb[p][q] = H1_.bb[p][q];
    ////        loop_mo_r loop_mo_s{
    ////            F_.aa[p][q] += V_.aaaa[p][r][q][s] * G1_.aa[s][r] + V_.abab[p][r][q][s] * G1_.bb[s][r];
    ////            F_.bb[p][q] += V_.bbbb[p][r][q][s] * G1_.bb[s][r] + V_.abab[r][p][s][q] * G1_.aa[s][r];
    ////        }
    ////    }
    
    ////    fprintf(outfile,"\n  The energy of the reference is: %20.12f Eh",E0_);
    ////    fprintf(outfile,"\n  Diagonal elements of the Fock matrix:");
    ////    fprintf(outfile,"\n  SO            Epsilon         ON");
    ////    loop_mo_p {
    ////        fprintf(outfile,"\n  %2d  %20.12f   %8.6f  %20.12f   %8.6f",p,F_.aa[p][p],G1_.aa[p][p],F_.bb[p][p],G1_.bb[p][p]);
    ////    }
    //}

}} // End Namespaces

#ifndef __LAPLACEDENOMINATOR_H__
#define __LAPLACEDENOMINATOR_H__

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

namespace forte {
class LaplaceDenominator {
    protected:
     // Fully split denominator (w in rows, i in columns)
     psi::SharedMatrix denominator_occ_;
     // Fully split denominator (w in rows, a in columns)
     psi::SharedMatrix denominator_vir_;
     // Fully split denominator (w in rows, u in columns)
     psi::SharedMatrix denominator_act_;

     // Pointer to active occupied orbital eigenvalues
     std::shared_ptr<psi::Vector> eps_occ_;
     // Pointer to active virtual orbital eigenvalues
     std::shared_ptr<psi::Vector> eps_vir_;
     // Pointer to active orbital eigenvalues
     std::shared_ptr<psi::Vector> eps_act_;
     // Number of vectors required to obtain given accuracy
     int nvector_;
     // Maximum error norm allowed in denominator
     double delta_;

     bool cavv_;

     void decompose_ccvv();

     void decompose_cavv();

     void decompose_ccav();

    public:
     LaplaceDenominator(std::shared_ptr<psi::Vector> eps_occ_, std::shared_ptr<psi::Vector> eps_vir, double delta);
     LaplaceDenominator(std::shared_ptr<psi::Vector> eps_occ, std::shared_ptr<psi::Vector> eps_act, std::shared_ptr<psi::Vector> eps_vir, double delta, bool cavv);
     ~LaplaceDenominator();
     psi::SharedMatrix denominator_occ() const { return denominator_occ_; }
     psi::SharedMatrix denominator_vir() const { return denominator_vir_; }
     psi::SharedMatrix denominator_act() const { return denominator_act_; }
};
}

#endif
/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _mcsrgpt2_mo_h_
#define _mcsrgpt2_mo_h_

#include <cmath>
#include <vector>

#include "psi4/libmints/vector.h"
#include "psi4/libmints/matrix.h"

#include "base_classes/rdms.h"
#include "base_classes/scf_info.h"
#include "helpers/timer.h"
#include "integrals/integrals.h"

#include "mrdsrg-helper/dsrg_source.h"

using d1 = std::vector<double>;
using d2 = std::vector<d1>;
using d3 = std::vector<d2>;
using d4 = std::vector<d3>;
using d5 = std::vector<d4>;
using d6 = std::vector<d5>;

namespace forte {

class MCSRGPT2_MO {
  public:
    /**
     * @brief The Constructor for the pilot DSRG-MRPT2 code
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    MCSRGPT2_MO(RDMs reference, std::shared_ptr<ForteOptions> options,
                std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~MCSRGPT2_MO();

    /// Compute the energy
    double compute_energy();

  protected:
    /// Source Operators
    enum sourceop { STANDARD, AMP, EMP2, LAMP, LEMP2 };
    std::map<std::string, sourceop> sourcemap = {
        {"STANDARD", STANDARD}, {"AMP", AMP}, {"EMP2", EMP2}, {"LAMP", LAMP}, {"LEMP2", LEMP2}};

    /// Basis preparation
    void startup();

    void cleanup();

    /// Integrals
    std::shared_ptr<ForteIntegrals> integral_;

    /// RDMs
    RDMs reference_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// ForteOptions
    std::shared_ptr<ForteOptions> options_;

    /// Printing level
    int print_;

    /// Molecular Orbitals
    size_t ncmo_; // correlated MOs
    size_t nfrzc_;
    size_t nfrzv_;
    size_t ncore_;
    std::vector<size_t> core_mos_;
    size_t nactv_;
    std::vector<size_t> actv_mos_;
    size_t nvirt_; // virtual MOs
    std::vector<size_t> virt_mos_;
    size_t nhole_; // hole MOs
    std::vector<size_t> hole_mos_;
    size_t npart_; // particle MOs
    std::vector<size_t> part_mos_;
    void prepare_mo_space();

    /// Symmetry
    std::vector<int> sym_actv_; // active MOs
    std::vector<int> sym_ncmo_; // correlated MOs

    /// DSRG s Parameter
    double s_;

    /// Source Operator
    std::string source_;

    /// Exponent of Delta
    double expo_delta_;

    /// Taylor Expansion Threshold
    int taylor_threshold_;
    int taylor_order_;

    /// RDMs Energy
    void compute_ref();

    /// Density Matrix
    d2 Da_;
    d2 Db_;

    /// 2-Body Density Cumulant
    d4 L2aa_;
    d4 L2ab_;
    d4 L2bb_;

    /// 3-Body Density Cumulant
    d6 L3aaa_;
    d6 L3aab_;
    d6 L3abb_;
    d6 L3bbb_;

    /// Fill in non-tensor cumulants used in the naive MR-DSRG-PT2 code
    void fill_naive_cumulants(RDMs ref, const int level);
    /// Fill in non-tensor quantities D1a_ and D1b_ using ambit tensors
    void fill_one_cumulant(ambit::Tensor& L1a, ambit::Tensor& L1b);
    /// Fill in non-tensor quantities L2aa_, L2ab_, and L2bb_ using ambit tensors
    void fill_two_cumulant(ambit::Tensor& L2aa, ambit::Tensor& L2ab, ambit::Tensor& L2bb);
    /// Fill in non-tensor quantities L3aaa_, L3aab_, L3abb_ and L3bbb_ using ambit tensors
    void fill_three_cumulant(ambit::Tensor& L3aaa, ambit::Tensor& L3aab, ambit::Tensor& L3abb,
                             ambit::Tensor& L3bbb);

    /// Print Density Matrix (Active ONLY)
    void print_density(const std::string& spin, const d2& density);
    /// Print 2-body cumulants
    void print2PDC(const std::string& str, const d4& TwoPDC, const int& PRINT);
    /// Print 3-body cumulants
    void print3PDC(const std::string& str, const d6& ThreePDC, const int& PRINT);

    /// Fock Matrix
    d2 Fa_;
    d2 Fb_;
    /// Form Fock matrix
    void Form_Fock(d2& A, d2& B);
    /// Compute Fock (stored in ForteIntegal) using this->Da_
    void compute_Fock_ints();
    /// Print Fock Matrix in Blocks
    void print_Fock(const std::string& spin, const d2& Fock);

    /// Form T Amplitudes for DSRG
    void Form_AMP_DSRG();

    /// T1 Amplitude
    d2 T1a_;
    d2 T1b_;
    double T1Na_;   // Norm T1a
    double T1Nb_;   // Norm T1b
    double T1Maxa_; // Max T1a
    double T1Maxb_; // Max T1b
    std::string t1_amp_;

    /// T2 Amplitude
    d4 T2aa_;
    d4 T2ab_;
    d4 T2bb_;
    double T2Naa_;   // Norm T2aa
    double T2Nab_;   // Norm T2ab
    double T2Nbb_;   // Norm T2bb
    double T2Maxaa_; // Max T2aa
    double T2Maxab_; // Max T2ab
    double T2Maxbb_; // Max T2bb

    /// Form T Amplitudes
    void Form_T2_DSRG(d4& AA, d4& AB, d4& BB, std::string& T_ALGOR);
    void Form_T1_DSRG(d2& A, d2& B);
    void Form_T2_ISA(d4& AA, d4& AB, d4& BB, const double& b_const);
    void Form_T1_ISA(d2& A, d2& B, const double& b_const);
    void Form_T2_SELEC(d4& AA, d4& AB, d4& BB);

    /// Check T Amplitudes
    void Check_T1(const std::string& x, const d2& M, double& Norm, double& MaxT,
                  std::shared_ptr<ForteOptions> options);
    void Check_T2(const std::string& x, const d4& M, double& Norm, double& MaxT,
                  std::shared_ptr<ForteOptions> options);

    /// Effective Fock Matrix
    d2 Fa_dsrg_;
    d2 Fb_dsrg_;
    void Form_Fock_DSRG(d2& A, d2& B, const bool& dsrgpt);

    /// Effective Two Electron Integral
    d4 vaa_dsrg_;
    d4 vab_dsrg_;
    d4 vbb_dsrg_;
    void Form_APTEI_DSRG(const bool& dsrgpt);

    /// Print Delta
    void PrintDelta();

    /// Computes the DSRG-MRPT2 energy
    double compute_energy_dsrg();

    /// Computes the SRG-MRPT2 energy
    double compute_energy_srg();

    /// Fock Matrix in SRG
    d2 Fa_srg_;
    d2 Fb_srg_;
    void Form_Fock_SRG();

    /// SRG-MRPT2 Energy Components
    double ESRG_11();
    double ESRG_12();
    double ESRG_21();
    double ESRG_22_2();
    double ESRG_22_4();
    double ESRG_22_6();

    /// SRG source operator, just need the linear one
    std::shared_ptr<DSRG_SOURCE> srg_source_;

    /// Energy Components
    void E_FT1(double& E);
    void E_VT1_FT2(double& EF1, double& EF2, double& EV1, double& EV2);
    void E_VT2_2(double& E);
    void E_VT2_4PP(double& E);
    void E_VT2_4HH(double& E);
    void E_VT2_4PH(double& E);
    void E_VT2_6(double& E1, double& E2);

    double Eref_;
    double Ecorr_;
    double Etotal_;

    /// Test denominators from Dyall / retaining excitation Hamiltonian
    void test_D1_RE();
    void test_D2_RE();
    void test_D2_Dyall();

    /// Timings
    void Print_Timing();
    local_timer dsrg_timer;
    double T2_timing;
    double T1_timing;
    double FT1_timing;
    double FT2_timing;
    double VT1_timing;
    double VT2C2_timing;
    double VT2C4_timing;
    double VT2C6_timing;

    /// Compute an addition element of renorm. H according to source operator
    double ElementRH(const std::string& source, const double& D, const double& V);

    /// Compute an element of T according to source operator
    double ElementT(const std::string& source, const double& D, const double& V);

    /// Taylor Expansion of [1 - exp(-|Z|^g)] / Z = Z^{g-1} \sum_{n=1}
    /// \frac{1}{n!} (-1)^{n+1} Z^{(n-1)g})
    double Taylor_Exp(const double& Z, const int& n, const double& g) {
        bool Znegative = Z < 0.0 ? 1 : 0;
        double Zcopy = Znegative ? -Z : Z;

        double value = 1, tmp = 1;
        for (int x = 0; x < (n - 1); ++x) {
            tmp *= -1.0 * pow(Zcopy, g) / (x + 2);
            value += tmp;
        }
        value *= pow(Zcopy, g - 1.0);
        return Znegative ? -value : value;
    }

    /// Taylor Expansion of [1 - exp(-|Z|)] / Z
    double Taylor_Exp_Linear(const double& Z, const int& n) {
        bool Zabs = Z > 0.0 ? 1 : 0;
        if (n > 0) {
            double value = 1, tmp = 1;
            for (int x = 0; x < (n - 1); ++x) {
                tmp *= pow(-1.0, Zabs) * Z / (x + 2);
                value += tmp;
            }
            return value * pow(-1.0, Zabs + 1);
        } else {
            return 0.0;
        }
    }
};
} // namespace forte

#endif // _mcsrgpt2_mo_h_

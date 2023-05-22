/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _rdms_h_
#define _rdms_h_

#include <string>
#include <vector>

namespace psi {
class Dimension;
class Matrix;
} // namespace psi

namespace ambit {
class Tensor;
}

namespace forte {

class ForteIntegrals;
class MOSpaceInfo;

/**
 * @class RDMs
 *
 * @brief This class stores diagonal or transition reduced density matrices (RDMs).
 *
 * The spin-orbital n-body reduced density matrix between two states |A> and |B> is defined as
 *
 *   RDM(p_1, p_2, ..., p_n, q_1, q_2, ..., q_n) = <A| a+(p1) ... a+(pn) a(qn) ... a(q1) |B>
 *
 * Two types of RDMs can be created: spin-dependent and spin-free.
 * The spin-dependent n-body RDMs are defined by
 *
 *   D(p1,σ1; ...; pn,σn; q1,σ1; ...; qn,σn) = <A| a+(p1,σ1) ... a+(pn,σn) a(qn,σn) ... a(q1,σ1) |B>
 *
 * where spins σ1 >= σ2 >= ... >= σn given that α > β.
 * The spin-free n-body RDMs are defined using spin-dependent RDMs as
 *
 *   F(p1, ..., pn, q1, ..., qn) = sum_{σ1,...,σn} D(p1,σ1; ...; pn,σn; q1,σ1; ...; qn,σn)
 *
 * This class is constructed by passing the RDMs up to a given rank n (n <= 3). For example,
 * to pass the spin-dependent one- and two-body RDMs an object is initialized as
 *
 * >>> ambit::Tensor g1a, g1b, g2aa, g2ab, g2bb;
 * >>> // ...
 * >>> // fill in g1a, g1b, ...
 * >>> // ...
 * >>> std::shared_ptr<RDMs> rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb);
 *
 * From the RDMs it is possible to obtain the corresponding cumulants by calling appropriate member
 * functions. For example, the alpha-alpha 2-body density cumulant can be obtained by calling
 *
 * >>> auto L2aa = rdms->L2aa();
 *
 * Given an orbital rotation matrix, the RDMs can be easily transformed to the new basis by calling
 *
 * >>> ambit::Tensor Ua, Ub; // orbital transformation for alpha and beta orbitals
 * >>> ... fill in Ua and Ub ...
 * >>> rdms->rotate(Ua, Ub);
 *
 * RDMs of the same spin type can be added together via AXPY
 *
 * >>> std::shared_ptr<RDMs> xrdms;
 * >>> xrdms = std::make_shared<RDMsSpinDependent>(xg1a, xg1b, xg2aa, xg2ab, xg2bb);
 * >>> rdm->axpy(xrdms, 0.5);
 *
 * which is equivalent to
 * >>> g1a("pq") += 0.5 * xg1a("pq");
 * >>> g1b("pq") += 0.5 * xg1b("pq");
 * >>> g2aa("pqrs") += 0.5 * xg2aa("pqrs");
 * >>> g2ab("pqrs") += 0.5 * xg2ab("pqrs");
 * >>> g2bb("pqrs") += 0.5 * xg2bb("pqrs");
 * >>> rdms = std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb);
 *
 * The RDMs can be dumped to disk by calling
 *
 * >>> rdms->dump_to_disk("casci.1a1");
 *
 * This will generate a series of files with names begining with "casci.1a1",
 * such as casci.1a1.g1a.bin, casci.1a1.g1b.bin, casci.1a1.g2aa.bin, ...
 * If these files are available, it is possible to initialize an RDMs object via
 *
 * >>> auto frdms = RDMs::build_from_disk(2, RDMsType::spin_dependent, "casci.1a1");
 *
 * which tries to read the following files: casci.1a1.g1a.bin, casci.1a1.g1b.bin,
 * casci.1a1.g2aa.bin, casci.1a1.g2ab.bin, and casci.1a1.g2ab.bin.
 * Note that names after the prefix (g1a.bin, g2aa.bin, ...) are hard coded.
 */

enum class RDMsType { spin_dependent, spin_free };

class RDMs {
  public:
    /// Build a zero-valued RDMs object
    static std::shared_ptr<RDMs> build(size_t max_rdm_level, size_t n_orbs, RDMsType type);
    /// Initialize RDMs from disk
    static std::shared_ptr<RDMs> build_from_disk(size_t max_rdm_level, RDMsType type,
                                                 const std::string& filename_prefix = "");

    /// Default desctructor
    virtual ~RDMs(){};

    /// @return the max RDM level
    size_t max_rdm_level() const { return max_rdm_; }

    /// @return RDM type
    RDMsType rdm_type() const { return type_; }

    /// @return dimension of each index
    size_t dim() const { return n_orbs_; }

    /// Clone the current RDMs
    virtual std::shared_ptr<RDMs> clone() = 0;

    /// Scale the current RDMs with a factor
    virtual void scale(double factor) = 0;

    /// AXPY: this += a * rhs + this
    virtual void axpy(std::shared_ptr<RDMs> rhs, double a) = 0;

    /// Rotate the current RDMs using the input unitary matrices
    virtual void rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) = 0;

    /// Save the current RDMs to disk (binary files)
    virtual void dump_to_disk(const std::string& filename_prefix = "") const = 0;
    /// Save the spin-summed 1-RDMs to disk in human readable form
    void save_SF_G1(const std::string& filename);

    // static methods

    /// Make alpha-alpha or beta-beta 2-RDC from 2-RDMs
    static ambit::Tensor make_cumulant_L2aa(const ambit::Tensor& g1a, const ambit::Tensor& g2aa);
    /// Make alpha-beta 2-RDC from 2-RDMs
    static ambit::Tensor make_cumulant_L2ab(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                            const ambit::Tensor& g2ab);
    /// Make alpha-alpha-alpha or beta-beta-beta 3-RDC from 3-RDMs
    static ambit::Tensor make_cumulant_L3aaa(const ambit::Tensor& g1a, const ambit::Tensor& g2aa,
                                             const ambit::Tensor& g3aaa);
    /// Make alpha-alpha-beta 3-RDC from 3-RDMs
    static ambit::Tensor make_cumulant_L3aab(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                             const ambit::Tensor& g2aa, const ambit::Tensor& g2ab,
                                             const ambit::Tensor& g3aab);
    /// Make alpha-beta-beta 3-RDC from 3-RDMs
    static ambit::Tensor make_cumulant_L3abb(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                             const ambit::Tensor& g2ab, const ambit::Tensor& g2bb,
                                             const ambit::Tensor& g3abb);

    /// Spin-free 1-body to spin-dependent 1-body subject to Ms averaging
    static ambit::Tensor sf1_to_sd1(const ambit::Tensor& G1);
    /// Spin-free 2-body to spin-dependent 2-body subject to Ms averaging
    static ambit::Tensor sf2_to_sd2aa(const ambit::Tensor& G2);
    static ambit::Tensor sf2_to_sd2ab(const ambit::Tensor& G2);
    /// Spin-free 3-body to spin-dependent 3-body subject to Ms averaging
    static ambit::Tensor sf3_to_sd3aaa(const ambit::Tensor& G3);
    static ambit::Tensor sf3_to_sd3aab(const ambit::Tensor& G3);
    static ambit::Tensor sf3_to_sd3abb(const ambit::Tensor& G3);

    // Spin-dependent reduced density matrices (RDMs)

    /// @return the alpha 1-RDM
    virtual ambit::Tensor g1a() const = 0;
    /// @return the beta 1-RDM
    virtual ambit::Tensor g1b() const = 0;
    /// @return the alpha-alpha 2-RDM
    virtual ambit::Tensor g2aa() const = 0;
    /// @return the alpha-beta 2-RDM
    virtual ambit::Tensor g2ab() const = 0;
    /// @return the beta-beta 2-RDM
    virtual ambit::Tensor g2bb() const = 0;
    /// @return the alpha-alpha-alpha 3-RDM
    virtual ambit::Tensor g3aaa() const = 0;
    /// @return the alpha-alpha-beta 3-RDM
    virtual ambit::Tensor g3aab() const = 0;
    /// @return the alpha-beta-beta 3-RDM
    virtual ambit::Tensor g3abb() const = 0;
    /// @return the beta-beta-beta 3-RDM
    virtual ambit::Tensor g3bbb() const = 0;

    // Spin-free RDMs

    /// @return the spin-free 1-RDM
    virtual ambit::Tensor SF_G1() const = 0;
    /// @return the spin-free 2-RDM
    virtual ambit::Tensor SF_G2() const = 0;
    /// @return the spin-free 3-RDM
    virtual ambit::Tensor SF_G3() const = 0;

    /// @return the spin-free 1-RDM in Psi4 Matrix format (nactv * nactv)
    std::shared_ptr<psi::Matrix> SF_G1mat();
    /// @return the spin-free 1-RDM in Psi4 Matrix format (nactvpi * nactvpi)
    std::shared_ptr<psi::Matrix> SF_G1mat(const psi::Dimension& dim);

    // Spin-dependent density cumulants

    /// @return the alpha 1-RDC
    virtual ambit::Tensor L1a() const = 0;
    /// @return the beta 1-RDC
    virtual ambit::Tensor L1b() const = 0;
    /// @return the alpha-alpha 2-RDC
    virtual ambit::Tensor L2aa() const = 0;
    /// @return the alpha-beta 2-RDC
    virtual ambit::Tensor L2ab() const = 0;
    /// @return the beta-beta 2-RDC
    virtual ambit::Tensor L2bb() const = 0;
    /// @return the alpha-alpha-alpha 3-RDC
    virtual ambit::Tensor L3aaa() const = 0;
    /// @return the alpha-alpha-beta 3-RDC
    virtual ambit::Tensor L3aab() const = 0;
    /// @return the alpha-beta-beta 3-RDC
    virtual ambit::Tensor L3abb() const = 0;
    /// @return the beta-beta-beta 3-RDC
    virtual ambit::Tensor L3bbb() const = 0;

    // Spin-free density cumulants

    /// @return the spin-free 1-cumulant
    ambit::Tensor SF_L1() const;
    /// @return the spin-free 2-cumulant
    ambit::Tensor SF_L2() const;
    /// @return the spin-free 3-cumulant
    ambit::Tensor SF_L3() const;

  protected:
    // ==> Class Data <==

    /// Maximum RDM/RDC rank stored by this object
    size_t max_rdm_ = 0;

    /// RDM type
    RDMsType type_;

    /// Number of orbitals for each dimension
    size_t n_orbs_ = 0;

    /// Test if the RDM dimensions are valid
    void _test_rdm_dims(const ambit::Tensor& T, const std::string& name,
                        size_t desired_dim_size) const;
    /// Test if a function is asked for the correct level of RDMs
    void _test_rdm_level(const size_t& level, const std::string& name) const;
    /// Test if RDMs rotation can be bypassed (i.e., if Ua and Ub are identity matrices)
    bool _bypass_rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub,
                        const double& zero_threshold = 1.0e-12) const;
};

class RDMsSpinDependent : public RDMs {
  public:
    /// @brief Default constructor
    RDMsSpinDependent();
    /// @brief Construct a RDMsSpinDependent object with the 1-rdm
    RDMsSpinDependent(ambit::Tensor g1a, ambit::Tensor g1b);
    /// @brief Construct a RDMsSpinDependent object with the 1- and 2-rdms
    RDMsSpinDependent(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
                      ambit::Tensor g2bb);
    /// @brief Construct a RDMsSpinDependent object with the 1-, 2-, and 3-rdms
    RDMsSpinDependent(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
                      ambit::Tensor g2bb, ambit::Tensor g3aaa, ambit::Tensor g3aab,
                      ambit::Tensor g3abb, ambit::Tensor g3bbb);

    /// @return the alpha 1-RDM
    ambit::Tensor g1a() const override;
    /// @return the beta 1-RDM
    ambit::Tensor g1b() const override;
    /// @return the alpha-alpha 2-RDM
    ambit::Tensor g2aa() const override;
    /// @return the alpha-beta 2-RDM
    ambit::Tensor g2ab() const override;
    /// @return the beta-beta 2-RDM
    ambit::Tensor g2bb() const override;
    /// @return the alpha-alpha-alpha 3-RDM
    ambit::Tensor g3aaa() const override;
    /// @return the alpha-alpha-beta 3-RDM
    ambit::Tensor g3aab() const override;
    /// @return the alpha-beta-beta 3-RDM
    ambit::Tensor g3abb() const override;
    /// @return the beta-beta-beta 3-RDM
    ambit::Tensor g3bbb() const override;

    // Spin-free RDMs

    /// @return the spin-free 1-RDM
    ambit::Tensor SF_G1() const override;
    /// @return the spin-free 2-RDM
    ambit::Tensor SF_G2() const override;
    /// @return the spin-free 3-RDM
    ambit::Tensor SF_G3() const override;

    // Spin-dependent density cumulants

    ambit::Tensor L1a() const override;
    /// @return the beta 1-RDC
    ambit::Tensor L1b() const override;
    /// @return the alpha-alpha 2-RDC
    ambit::Tensor L2aa() const override;
    /// @return the alpha-beta 2-RDC
    ambit::Tensor L2ab() const override;
    /// @return the beta-beta 2-RDC
    ambit::Tensor L2bb() const override;
    /// @return the alpha-alpha-alpha 3-RDC
    ambit::Tensor L3aaa() const override;
    /// @return the alpha-alpha-beta 3-RDC
    ambit::Tensor L3aab() const override;
    /// @return the alpha-beta-beta 3-RDC
    ambit::Tensor L3abb() const override;
    /// @return the beta-beta-beta 3-RDC
    ambit::Tensor L3bbb() const override;

    // class methods

    /// Clone the current RDMs
    std::shared_ptr<RDMs> clone() override;

    /// Scale the current RDMs with a factor
    void scale(double factor) override;

    /// AXPY: this += a * rhs + this
    void axpy(std::shared_ptr<RDMs> rhs, double a) override;

    /// Rotate the current RDMs using the input unitary matrices
    void rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) override;

    /// Save the current RDMs to disk
    void dump_to_disk(const std::string& filename_prefix = "") const override;

  private:
    /// The alpha 1-RDM
    ambit::Tensor g1a_;
    /// The beta 1-RDM
    ambit::Tensor g1b_;
    /// The alpha-alpha 2-RDM
    ambit::Tensor g2aa_;
    /// The alpha-beta 2-RDM
    ambit::Tensor g2ab_;
    /// The beta-beta 2-RDM
    ambit::Tensor g2bb_;
    /// The alpha-alpha-alpha 3-RDM
    ambit::Tensor g3aaa_;
    /// The alpha-alpha-beta 3-RDM
    ambit::Tensor g3aab_;
    /// The alpha-beta-beta 3-RDM
    ambit::Tensor g3abb_;
    /// The beta-beta-beta 3-RDM
    ambit::Tensor g3bbb_;
};

class RDMsSpinFree : public RDMs {
  public:
    /// @brief Default constructor
    RDMsSpinFree();
    /// @brief Construct a RDMsSpinFree object with the 1-rdm
    RDMsSpinFree(ambit::Tensor G1);
    /// @brief Construct a RDMsSpinFree object with the 1- and 2-rdms
    RDMsSpinFree(ambit::Tensor G1, ambit::Tensor G2);
    /// @brief Construct a RDMsSpinFree object with the 1-, 2-, and 3-rdms
    RDMsSpinFree(ambit::Tensor G1, ambit::Tensor G2, ambit::Tensor G3);

    /// @return the alpha 1-RDM
    ambit::Tensor g1a() const override;
    /// @return the beta 1-RDM
    ambit::Tensor g1b() const override;
    /// @return the alpha-alpha 2-RDM
    ambit::Tensor g2aa() const override;
    /// @return the alpha-beta 2-RDM
    ambit::Tensor g2ab() const override;
    /// @return the beta-beta 2-RDM
    ambit::Tensor g2bb() const override;
    /// @return the alpha-alpha-alpha 3-RDM
    ambit::Tensor g3aaa() const override;
    /// @return the alpha-alpha-beta 3-RDM
    ambit::Tensor g3aab() const override;
    /// @return the alpha-beta-beta 3-RDM
    ambit::Tensor g3abb() const override;
    /// @return the beta-beta-beta 3-RDM
    ambit::Tensor g3bbb() const override;

    // Spin-free RDMs

    /// @return the spin-free 1-RDM
    ambit::Tensor SF_G1() const override;
    /// @return the spin-free 2-RDM
    ambit::Tensor SF_G2() const override;
    /// @return the spin-free 3-RDM
    ambit::Tensor SF_G3() const override;

    // Spin-dependent density cumulants

    ambit::Tensor L1a() const override;
    /// @return the beta 1-RDC
    ambit::Tensor L1b() const override;
    /// @return the alpha-alpha 2-RDC
    ambit::Tensor L2aa() const override;
    /// @return the alpha-beta 2-RDC
    ambit::Tensor L2ab() const override;
    /// @return the beta-beta 2-RDC
    ambit::Tensor L2bb() const override;
    /// @return the alpha-alpha-alpha 3-RDC
    ambit::Tensor L3aaa() const override;
    /// @return the alpha-alpha-beta 3-RDC
    ambit::Tensor L3aab() const override;
    /// @return the alpha-beta-beta 3-RDC
    ambit::Tensor L3abb() const override;
    /// @return the beta-beta-beta 3-RDC
    ambit::Tensor L3bbb() const override;

    // class methods

    /// Clone the current RDMs
    std::shared_ptr<RDMs> clone() override;

    /// Scale the current RDMs with a factor
    void scale(double factor) override;

    /// AXPY: this += a * rhs + this
    void axpy(std::shared_ptr<RDMs> rhs, double a) override;

    /// Rotate the current RDMs using the input unitary matrices
    void rotate(const ambit::Tensor& Ua, const ambit::Tensor& Ub) override;

    /// Save the current RDMs to disk
    void dump_to_disk(const std::string& filename_prefix = "") const override;

  private:
    /// Spin-free (spin-summed) 1-RDM defined as G1[pq] = g1a[pq] + g1b[pq]
    ambit::Tensor SF_G1_;
    /// Spin-free (spin-summed) 2-RDM defined as
    /// G2[pqrs] = g2aa[pqrs] + g2ab[pqrs] + g2ab[qpsr] + g2bb[pqrs]
    ambit::Tensor SF_G2_;
    /// Spin-free (spin-summed) 3-RDMs defined as G3[pqrstu] = g3aaa[pqrstu] + g3aab[pqrstu] +
    /// g3aab[prqsut] + g3aab[qrptus] + g3abb[pqrstu] + g3abb[qprtsu] + g3abb[rpqust]
    ambit::Tensor SF_G3_;
};
} // namespace forte

#endif // _rdms_h_

#ifndef _integrals_h_
#define _integrals_h_

#define PAIRINDEX(i,j) ((i>j) ? (ioff[(i)]+(j)) : (ioff[(j)]+(i)))
#define four(i,j,k,l) PAIRINDEX(PAIRINDEX(i,j),PAIRINDEX(k,l))

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/dynamic_bitset.hpp>

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <libtrans/integraltransform.h>
#include <libmints/matrix.h>
#include <libthce/thce.h>
#include <ambit/blocked_tensor.h>

namespace psi{ namespace libadaptive{

/// This decides the type of transformation: resticted vs. unrestricted
enum IntegralSpinRestriction {RestrictedMOs,UnrestrictedMOs};
enum IntegralFrozenCore {RemoveFrozenMOs,KeepFrozenMOs};
enum IntegralType {ConventionalInts,DFInts,CholeskyInts};

/**
 * Integrals: transforms and stores the integrals in Pitzer ordering
 */
class ExplorerIntegrals{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param options The main options object
     * @param restricted Select a restricted or unrestricted transformation (RestrictedMOs = restricted, UnrestrictedMOs = unrestricted).
     * @param resort_frozen_core Determines if the the integral with frozen index are removed
     *        (RemoveFrozenMOs = remove and resort, KeepFrozenMOs = keep all the integrals).
     */
    ExplorerIntegrals(psi::Options &options,IntegralSpinRestriction restricted,IntegralFrozenCore resort_frozen_core);

    /// Destructor
    virtual ~ExplorerIntegrals();


    // ==> Class Interface <==

    /// Return the total number of molecular orbitals (this number includes frozen MOs)
    size_t nmo() const {return nmo_;}
    virtual size_t nthree() const = 0;
    
    /// Return the number of irreducible representations
    int nirrep() const {return nirrep_;}

    /// Return the total number of correlated molecular orbitals (this number excludes frozen MOs)
    size_t ncmo() const {return ncmo_;}

    /// Return the number of frozen core orbitals per irrep
    Dimension& frzcpi() {return frzcpi_;}
    /// Return the number of frozen virtual orbitals per irrep
    Dimension& frzvpi() {return frzvpi_;}

    /// The number of correlated MOs per irrep (non frozen).  This is nmopi - nfzcpi - nfzvpi.
    Dimension& ncmopi() {return ncmopi_;}

    /// Return the frozen core energy
    double frozen_core_energy() {return frozen_core_energy_;}

    /// Scalar component of the Hamiltonian
    double scalar() const {return scalar_;}

    /// The alpha one-electron integrals
    double oei_a(size_t p,size_t q) {return one_electron_integrals_a[p * aptei_idx_ + q];}

    /// The beta one-electron integrals
    double oei_b(size_t p,size_t q) {return one_electron_integrals_b[p * aptei_idx_ + q];}

    /// The diagonal fock matrix integrals
    double fock_a(size_t p,size_t q) {return fock_matrix_a[p * aptei_idx_ + q];}

    /// The diagonal fock matrix integrals
    double fock_b(size_t p,size_t q) {return fock_matrix_b[p * aptei_idx_ + q];}

    /// The diagonal fock matrix integrals
    double diag_fock_a(size_t p) {return fock_matrix_a[p * aptei_idx_ + p];}

    /// The diagonal fock matrix integrals
    double diag_fock_b(size_t p) {return fock_matrix_b[p * aptei_idx_ + p];}

    /// The antisymmetrixed alpha-alpha two-electron integrals in physicist notation <pq||rs>
    virtual double aptei_aa(size_t p,size_t q,size_t r, size_t s) = 0;

    /// The antisymmetrixed alpha-beta two-electron integrals in physicist notation <pq||rs>
    virtual double aptei_ab(size_t p,size_t q,size_t r, size_t s) = 0;
    /// The antisymmetrixed beta-beta two-electron integrals in physicist notation <pq||rs>
    virtual double aptei_bb(size_t p,size_t q,size_t r, size_t s) = 0;
    
    /// Reads the antisymmetrized alpha-alpha chunck and returns an ambit::Tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s) = 0;
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s) = 0;
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s) = 0;

    virtual double get_three_integral(size_t A, size_t p, size_t q) = 0;
    virtual ambit::Tensor get_three_integral_block(const std::vector<size_t>& A, const std::vector<size_t>& p, const std::vector<size_t>& q) = 0;

    /// The diagonal antisymmetrixed alpha-alpha two-electron integrals in physicist notation <pq||pq>

    virtual double diag_aptei_aa(size_t p,size_t q) = 0;

    /// The diagonal antisymmetrixed alpha-beta two-electron integrals in physicist notation <pq||pq>
    virtual double diag_aptei_ab(size_t p,size_t q) = 0;

    /// The diagonal antisymmetrixed beta-beta two-electron integrals in physicist notation <pq||pq>
    virtual double diag_aptei_bb(size_t p,size_t q) = 0;

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(SharedMatrix gamma_a,SharedMatrix gamma_b) = 0;

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(bool* Ia, bool* Ib) = 0;

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib) = 0;

    /// Make the diagonal matrix elements of the Fock operator for a given set of occupation numbers
    virtual void make_fock_diagonal(bool* Ia, bool* Ib,std::pair<std::vector<double>,std::vector<double> >& fock_diagonals) = 0;
    virtual void make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals) = 0;
    virtual void make_beta_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals) = 0;

    /// Set the value of the scalar part of the Hamiltonian
    /// @param value the new value of the scalar part of the Hamiltonian
    void set_scalar(double value) {scalar_ = value;}

    /// Set the value of the one-electron integrals
    /// @param ints pointer to the integrals
    /// @param the spin type of the integrals
    void set_oei(double** ints,bool alpha);

    /// Set the value of the one-electron integrals
    /// @param the spin type of the integrals
    void set_oei(size_t p, size_t q,double value,bool alpha);

    /// Set the value of the two-electron integrals
    /// @param ints pointer to the integrals
    /// @param the spin type of the integrals

    /// Set the value of the two-electron integrals
    /// @param the spin type of the integrals
    virtual void set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2) = 0;


    /// Update all integrals after providing one- and two-electron integrals
    /// via the functions set_oei and set_tei
    /// Sample use:
    ///     ExplorerIntegrals* ints = new ...
    ///
    ///     // One-electron integrals are updated
    ///     ints->set_oei(oei_aa,true);
    ///     ints->set_oei(oei_bb,false);
    ///     ints->update_integrals();
    virtual void update_integrals(bool freeze_core = true) = 0;

    /// Update the integrals with a new set of MO coefficients
    virtual void retransform_integrals() = 0;
    virtual double** get_three_integral_pointer() = 0;

    /// Get the fock matrix elements
    double get_fock_a(size_t p, size_t q){return fock_matrix_a[p * aptei_idx_ + q];}
    double get_fock_b(size_t p, size_t q){return fock_matrix_b[p * aptei_idx_ + q];}
    std::vector<size_t> get_cmotomo(){return cmotomo_;}


protected:

    // ==> Class data <==

    /// The options object
    psi::Options& options_;
    /// The integral_type
    std::string integral_type_;

    /// Are we doing a spin-restricted computation?
    IntegralSpinRestriction restricted_;

    /// Do we have to resort the integrals to eliminate frozen orbitals?
    IntegralFrozenCore resort_frozen_core_;

    /// Number of irreps
    size_t nirrep_;

    /// The number of MOs, including the ones that are frozen.
    size_t nmo_;

    /// The number of auxiliary basis functions if DF
    /// The number of cholesky vectors
    /// The number of cholesky/auxiliary
    /// Three Index Shared Matrix
    /// The number of correlated MOs (excluding frozen).  This is nmo - nfzc - nfzv.
    size_t ncmo_;

    /// The number of MOs per irrep.
    Dimension nmopi_;
    /// The number of frozen core MOs per irrep.
    Dimension frzcpi_;
    /// The number of frozen unoccupied MOs per irrep.
    Dimension frzvpi_;
    /// The number of correlated MOs per irrep (non frozen).  This is nmopi - nfzcpi - nfzvpi.
    Dimension ncmopi_;

    size_t aptei_idx_;
    size_t nso_;
    static bool have_omp_;
    int num_threads_;

    /// Number of one electron integrals
    size_t num_oei;

    /// Number of two electron integrals in chemist notation (pq|rs)
    size_t num_tei;

    /// The number of antisymmetrized two-electron integrals in physicist notation <pq||rs>
    size_t num_aptei;

    /// Frozen-core energy
    double frozen_core_energy_;
    std::vector<int> pair_irrep_map;
    std::vector<int> pair_index_map;

    double scalar_;

    /// One-electron integrals stored as a vector
    double* one_electron_integrals_a;
    double* one_electron_integrals_b;

    double* fock_matrix_a;
    double* fock_matrix_b;

    // ==> Class private functions <==

    void startup();

    void transform_one_electron_integrals();

    void resort_two(double*& ints,std::vector<size_t>& map);

    // ==> Class private virtual functions <==

    /// Allocate memory
    virtual void allocate();

    /// Deallocate memory
    virtual void deallocate();

    virtual void make_diagonal_integrals() = 0;

    /// This function manages freezing core and virtual orbitals
    virtual void freeze_core_orbitals() = 0;

    /// Compute the frozen core energy
    virtual void compute_frozen_core_energy() = 0;

    /// Compute the one-body operator modified by the frozen core orbitals
    virtual void compute_frozen_one_body_operator() = 0;

    /// Remove the doubly occupied and virtual orbitals and resort the rest so that
    /// we are left only with ncmo = nmo - nfzc - nfzv
    virtual void resort_integrals_after_freezing() = 0;


    virtual void resort_three(boost::shared_ptr<Matrix>&, std::vector<size_t>& map) = 0;
    virtual void resort_four(double*& tei, std::vector<size_t>& map) = 0;

    /// Look at CD/DF/Conventional to see implementation
    /// computes/reads integrals
    virtual void gather_integrals() = 0;
    /// The B tensor
    boost::shared_ptr<psi::Tensor> B_;
    /// The mapping from correlated to actual MO
    /// Basically gives the original ordering back
    std::vector<size_t> cmotomo_;
    /// The type of tensor that ambit uses -> kCore
    ambit::TensorType tensor_type_ = ambit::kCore;
    /// How much memory each integral takes up
    double int_mem_;
};

/**
 * @brief The ConventionalIntegrals class is an interface to calculate the conventional integrals
 * Assumes storage of all tei and stores in core.
 */
class ConventionalIntegrals: public ExplorerIntegrals{
public:
    ///Contructor of the class.  Calls ExplorerIntegrals constructor
    ConventionalIntegrals(psi::Options &options,IntegralSpinRestriction restricted,IntegralFrozenCore resort_frozen_core);
    virtual ~ConventionalIntegrals();

    /// Grabs the antisymmetriced TEI - assumes storage in aphy_tei_*
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Grabs the antisymmetrized TEI - assumes storage of ambit tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);

    virtual double diag_aptei_aa(size_t p, size_t q){return diagonal_aphys_tei_aa[p * aptei_idx_ + q];}
    virtual double diag_aptei_ab(size_t p, size_t q){return diagonal_aphys_tei_ab[p * aptei_idx_ + q];}
    virtual double diag_aptei_bb(size_t p, size_t q){return diagonal_aphys_tei_bb[p * aptei_idx_ + q];}
    virtual void retransform_integrals();
    virtual void update_integrals(bool freeze_core = true);
    virtual double get_three_integral(size_t A, size_t p, size_t q)
    {
        outfile->Printf("\n Oh no!, you tried to grab a ThreeIntegral but this is not there!!");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral");
    }
    virtual ambit::Tensor get_three_integral_block(const std::vector<size_t>& A, const std::vector<size_t>& p, const std::vector<size_t>& q)
    {
        outfile->Printf("\n Oh no!, you tried to grab a ThreeIntegral but this is not there!!");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral");
    }
    virtual double** get_three_integral_pointer()
    {
        outfile->Printf("\n Doh! There is no Three_integral here.  Use DF/CD");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral!");
    }

    virtual void make_fock_matrix(SharedMatrix gamma_a,SharedMatrix gamma_b);

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(bool* Ia, bool* Ib);

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib);

    /// Make the diagonal matrix elements of the Fock operator for a given set of occupation numbers
    virtual void make_fock_diagonal(bool* Ia, bool* Ib,std::pair<std::vector<double>,std::vector<double> >& fock_diagonals);
    virtual void make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    virtual void make_beta_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    virtual size_t nthree() const

    {
        throw PSIEXCEPTION("Wrong Int_Type");
    }

private:
    /// Transform the integrals
    void transform_integrals();

    virtual void gather_integrals();
    //Allocates memory for a antisymmetriced tei (nmo_^4)
    virtual void allocate();
    virtual void deallocate();
    //Calculates the diagonal integrals from aptei
    virtual void make_diagonal_integrals();
    virtual void freeze_core_orbitals();
    virtual void compute_frozen_core_energy();
    virtual void compute_frozen_one_body_operator();
    virtual void resort_integrals_after_freezing();
    virtual void resort_four(double*& tei, std::vector<size_t>& map);
    virtual void resort_three(boost::shared_ptr<Matrix>& threeint, std::vector<size_t>& map){}
    virtual void set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2);

    /// An addressing function to retrieve the two-electron integrals
    size_t aptei_index(size_t p,size_t q,size_t r,size_t s) {return aptei_idx_ * aptei_idx_ * aptei_idx_ * p + aptei_idx_ * aptei_idx_ * q + aptei_idx_ * r + s;}
    /// Function to get three index integral

    /// The IntegralTransform object used by this class
    IntegralTransform* ints_;

    double* aphys_tei_aa;
    double* aphys_tei_ab;
    double* aphys_tei_bb;
    double* diagonal_aphys_tei_aa;
    double* diagonal_aphys_tei_ab;
    double* diagonal_aphys_tei_bb;
};


/// Classes written by Kevin Hannon
///
/**
 * @brief The CholeskyIntegrals:  An interface that computes the cholesky integrals,
 * freezes the core, and creates fock matrices from determinant classes
 */
class CholeskyIntegrals : public ExplorerIntegrals{
public:
    CholeskyIntegrals(psi::Options &options,IntegralSpinRestriction restricted,IntegralFrozenCore resort_frozen_core);
    virtual ~CholeskyIntegrals();
    ///aptei_x will grab antisymmetriced integrals and creates DF/CD integrals on the fly
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Grabs the antisymmetrized TEI - assumes storage of ambit tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);

    virtual double diag_aptei_aa(size_t p, size_t q){return diagonal_aphys_tei_aa[p * aptei_idx_ + q];}
    virtual double diag_aptei_ab(size_t p, size_t q){return diagonal_aphys_tei_ab[p * aptei_idx_ + q];}
    virtual double diag_aptei_bb(size_t p, size_t q){return diagonal_aphys_tei_bb[p * aptei_idx_ + q];}
    virtual double get_three_integral(size_t A, size_t p, size_t q){return ThreeIntegral_->get(A,p * aptei_idx_ + q);}
    virtual double** get_three_integral_pointer(){return ThreeIntegral_->pointer();}
    virtual ambit::Tensor get_three_integral_block(const std::vector<size_t> &A, const std::vector<size_t> &p, const std::vector<size_t> &q);
    virtual void retransform_integrals();
    virtual void update_integrals(bool freeze_core = true);
    ///Do not use this if you are using CD/DF integrals
    virtual void set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2);

    virtual void make_fock_matrix(SharedMatrix gamma_a,SharedMatrix gamma_b);
    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(bool* Ia, bool* Ib);

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib);

    /// Make the diagonal matrix elements of the Fock operator for a given set of occupation numbers
    virtual void make_fock_diagonal(bool* Ia, bool* Ib,std::pair<std::vector<double>,std::vector<double> >& fock_diagonals);
    virtual void make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    virtual void make_beta_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    virtual size_t nthree() const {return nthree_;}

private:
    ///Computes Cholesky integrals
    virtual void gather_integrals();
    ///Allocates diagonal integrals
    virtual void allocate();
    virtual void deallocate();
    virtual void make_diagonal_integrals();
    virtual void freeze_core_orbitals();
    virtual void compute_frozen_core_energy();
    virtual void compute_frozen_one_body_operator();
    virtual void resort_three(boost::shared_ptr<Matrix>& threeint, std::vector<size_t>& map);
    virtual void resort_integrals_after_freezing();
    ///This is not used in Cholesky, but I have to have implementations for
    /// derived classes.
    virtual void resort_four(double *&tei, std::vector<size_t> &map)
    {
        outfile->Printf("If this is called, sig fault!");
        throw PSIEXCEPTION("No four integrals to sort");

    }
    boost::shared_ptr<Matrix> ThreeIntegral_;
    double* diagonal_aphys_tei_aa;
    double* diagonal_aphys_tei_ab;
    double* diagonal_aphys_tei_bb;
    size_t nthree_;
};


/**
 * @brief The DFIntegrals class - interface to get DF integrals, freeze core and resort,
 * make fock matrices, and grab information about the space
 */
class DFIntegrals : public ExplorerIntegrals{
public:
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Reads the antisymmetrized alpha-alpha chunck and returns an ambit::Tensor
    /// Grabs the antisymmetrized TEI - assumes storage of ambit tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r, 
        const std::vector<size_t>& s);

    virtual double diag_aptei_aa(size_t p, size_t q){return diagonal_aphys_tei_aa[p * aptei_idx_ + q];}
    virtual double diag_aptei_ab(size_t p, size_t q){return diagonal_aphys_tei_ab[p * aptei_idx_ + q];}
    virtual double diag_aptei_bb(size_t p, size_t q){return diagonal_aphys_tei_bb[p * aptei_idx_ + q];}
    virtual double get_three_integral(size_t A, size_t p, size_t q){return ThreeIntegral_->get(p * aptei_idx_ + q, A);}
    virtual ambit::Tensor get_three_integral_block(const std::vector<size_t>& A, const std::vector<size_t>& p, const std::vector<size_t>& q);
    virtual double** get_three_integral_pointer(){return ThreeIntegral_->pointer();}
    virtual void retransform_integrals();
    virtual void update_integrals(bool freeze_core = true);
    virtual void set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2);
    DFIntegrals(psi::Options &options,IntegralSpinRestriction restricted,IntegralFrozenCore resort_frozen_core);
    virtual ~DFIntegrals();

    virtual void make_fock_matrix(SharedMatrix gamma_a,SharedMatrix gamma_b);

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(bool* Ia, bool* Ib);

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib);

    /// Make the diagonal matrix elements of the Fock operator for a given set of occupation numbers
    virtual void make_fock_diagonal(bool* Ia, bool* Ib,std::pair<std::vector<double>,std::vector<double> >& fock_diagonals);
    virtual void make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    virtual void make_beta_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    virtual size_t nthree() const {return nthree_;}
private:
    virtual void gather_integrals();
    virtual void allocate();
    virtual void deallocate();
    //Grabs DF integrals with new Ca coefficients
    virtual void make_diagonal_integrals();
    virtual void freeze_core_orbitals();
    virtual void compute_frozen_core_energy();
    virtual void compute_frozen_one_body_operator();
    virtual void resort_three(boost::shared_ptr<Matrix>& threeint, std::vector<size_t>& map);
    virtual void resort_integrals_after_freezing();
    virtual void resort_four(double *&tei, std::vector<size_t> &map){}

    boost::shared_ptr<Matrix> ThreeIntegral_;
    double* diagonal_aphys_tei_aa;
    double* diagonal_aphys_tei_ab;
    double* diagonal_aphys_tei_bb;
    size_t nthree_;
    };
class DISKDFIntegrals : public ExplorerIntegrals{
public:
    DISKDFIntegrals(psi::Options &options,IntegralSpinRestriction restricted,IntegralFrozenCore resort_frozen_core);

    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Reads the antisymmetrized alpha-alpha chunck and returns an ambit::Tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
        const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r, 
        const std::vector<size_t>& s);

    virtual double diag_aptei_aa(size_t p, size_t q){return diagonal_aphys_tei_aa[p * aptei_idx_ + q];}
    virtual double diag_aptei_ab(size_t p, size_t q){return diagonal_aphys_tei_ab[p * aptei_idx_ + q];}
    virtual double diag_aptei_bb(size_t p, size_t q){return diagonal_aphys_tei_bb[p * aptei_idx_ + q];}
    virtual double get_three_integral(size_t A, size_t p, size_t q);
    virtual double** get_three_integral_pointer()
    {
        return (ThreeIntegral_->pointer());
    }
    ///Read a block of the DFIntegrals and return an Ambit tensor of size A by p by q
    virtual ambit::Tensor get_three_integral_block(const std::vector<size_t>& A, const std::vector<size_t>& p, const std::vector<size_t>& q);

    virtual void retransform_integrals();
    virtual void update_integrals(bool freeze_core = true);
    virtual void set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2);
    virtual ~DISKDFIntegrals();

    virtual void make_fock_matrix(SharedMatrix gamma_a,SharedMatrix gamma_b);

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(bool* Ia, bool* Ib);

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib);

    /// Make the diagonal matrix elements of the Fock operator for a given set of occupation numbers
    virtual void make_fock_diagonal(bool* Ia, bool* Ib,std::pair<std::vector<double>,std::vector<double> >& fock_diagonals);
    virtual void make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    virtual void make_beta_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    virtual size_t nthree() const {return nthree_;}
private:
    virtual void gather_integrals();
    virtual void allocate();
    virtual void deallocate();
    //Grabs DF integrals with new Ca coefficients
    virtual void make_diagonal_integrals();
    virtual void freeze_core_orbitals();
    virtual void compute_frozen_core_energy();
    virtual void compute_frozen_one_body_operator();
    virtual void resort_three(boost::shared_ptr<Matrix>& threeint, std::vector<size_t>& map);
    virtual void resort_integrals_after_freezing();
    virtual void resort_four(double *&tei, std::vector<size_t> &map){}

    boost::shared_ptr<Matrix> ThreeIntegral_;
    double* diagonal_aphys_tei_aa;
    double* diagonal_aphys_tei_ab;
    double* diagonal_aphys_tei_bb;
    size_t nthree_;
};


}} // End Namespaces

#endif // _integrals_h_


#ifndef _fci_solver_h_
#define _fci_solver_h_

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <physconst.h>

#include "fci_vector.h"

#include "helpers.h"
#include "integrals.h"
#include "string_lists.h"
#include "reference.h"


namespace psi{ namespace forte{

/**
 * @brief The FCISolver class
 * This class performs Full CI calculations.
 */
class FCISolver
{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * @brief FCISolver
     * @param active_dim The dimension of the active orbital space
     * @param core_mo A vector of doubly occupied orbitals
     * @param active_mo A vector of active orbitals
     * @param na Number of alpha electrons
     * @param nb Number of beta electrons
     * @param multiplicity The spin multiplicity (2S + 1).  1 = singlet, 2 = doublet, ...
     * @param symmetry The irrep of the FCI wave function
     * @param ints An integral object
     * @param mo_space_info -> MOSpaceInfo
     * @param initial_guess_per_root get from options object
     * @param print Control printing of FCISolver
     */
    FCISolver(Dimension active_dim, std::vector<size_t> core_mo,
              std::vector<size_t> active_mo, size_t na, size_t nb,
              size_t multiplicity, size_t symmetry, std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
              size_t initial_guess_per_root, int print, Options& options);
    /**
     * @brief FCISolver
     * @param active_dim The dimension of the active orbital space
     * @param core_mo A Vector of doubly occupied orbitals
     * @param active_mo A vector of active orbitals
     * @param na Number of alpha electrons
     * @param nb Number of beta electrons
     * @param multiplicity The spin multiplicity (2S + 1)
     * @param symmetry The Irrep of the FCI wave function
     * @param ints An integral object
     * @param mo_space_info -> mo_space_info object
     * @param options object
     */
    FCISolver(Dimension active_dim, std::vector<size_t> core_mo,
              std::vector<size_t> active_mo, size_t na, size_t nb,
              size_t multiplicity, size_t symmetry, std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
              Options& options);

    ~FCISolver() {}

    /// Compute the FCI energy
    double compute_energy();

    /// Return a reference object
    Reference reference();

    /// Set the number of desired roots
    void set_nroot(int value);
    /// Set the root that will be used to compute the properties
    void set_root(int value);
    /// Set the maximum RDM computed (0 - 3)
    void set_max_rdm_level(int value);
    /// Set the convergence for FCI
    void set_fci_iterations(int value);
    /// Set the number of collapse vectors for each root
    void set_collapse_per_root(int value);
    /// Set the maximum subspace size for each root
    void set_subspace_per_root(int value);
    /// Use a JK builder for the restricted_docc
    /// If you actually change the integrals in your code, you should set this to false.
    void set_use_jk_builder(bool jk_build);



    /// When set to true before calling compute_energy(), it will test the
    /// reduce density matrices.  Watch out, this function is very slow!
    void test_rdms(bool value) {test_rdms_ = value;}
    /// Print the Natural Orbitals
    void print_no(bool value){print_no_ = value;}
private:

    // ==> Class Data <==

    /// The Dimension object for the active space
    Dimension active_dim_;

    /// The orbitals frozen at the CI level
    std::vector<size_t> core_mo_;

    /// The orbitals treated at the CI level
    std::vector<size_t> active_mo_;

    /// A object that stores string information
    std::shared_ptr<StringLists> lists_;

    /// The molecular integrals
    std::shared_ptr<ForteIntegrals>  ints_;

    /// The FCI energy
    double energy_;

    /// The FCI wave function
    std::shared_ptr<FCIWfn> C_;

    /// The number of irreps
    int nirrep_;
    /// The symmetry of the wave function
    int symmetry_;
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The multiplicity (2S + 1) of the state to target.
    /// (1 = singlet, 2 = doublet, 3 = triplet, ...)
    int multiplicity_;
    /// The number of roots (default = 1)
    int nroot_ = 1;
    /// The root used to compute properties (zero based, default = 0)
    int root_ = 0;
    /// The number of trial guess vectors to generate per root
    size_t ntrial_per_root_; 
    /// The number of collapse vectors for each root
    size_t collapse_per_root_ = 2;
    /// The maximum subspace size for each root
    size_t subspace_per_root_ = 4;
    /// The maximum RDM computed (0 - 3)
    int max_rdm_level_;
    /// Iterations for FCI
    int fci_iterations_ = 30;
    /// Test the RDMs?
    bool test_rdms_ = false;
    /// Print the NO from the 1-RDM
    bool print_no_ = false;
    /// A variable to control printing information
    int print_ = 0;
    bool use_jk_ = true;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();
    /// The mo_space_info object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Initial CI wave function guess
    std::vector<std::pair<int,std::vector<std::tuple<size_t,size_t,size_t,double>>>>
    initial_guess(FCIWfn& diag,size_t n,size_t multiplicity,
                  std::shared_ptr<FCIIntegrals> fci_ints);
    /// The options object
    Options& options_;
};


/**
 * @brief The FCI class
 * This class implements a FCI wave function and calls FCISolver
 */
class FCI : public Wavefunction
{
public:

    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param wfn The main wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    FCI(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~FCI();

    // ==> Class Interface <==

    /// Compute the energy
    virtual double compute_energy();
    /// Return a reference object
    Reference reference();
    /// Set the print level
    void set_print(int value) {print_ = value;}
    /// Set the maximum RDM computed (0 - 3)
    void set_max_rdm_level(int value);
    /// FCI  iterations
    void set_fci_iterations(int value);
    /// Print the NO from the 1RDM
    void print_no(bool value);

private:

    // ==> Class data <==

    /// A reference to the options object
    Options& options_;
    /// The molecular integrals
    std::shared_ptr<ForteIntegrals>  ints_;
    /// The information about the molecular orbital spaces
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// A pointer to the FCISolver object
    FCISolver* fcisolver_ = nullptr;
    /// Print level
    /// 0 : silent mode (no printing)
    /// 1 : default printing
    int print_  = 1;

    /// Set the maximum RDM computed (0 - 3)
    int max_rdm_level_;
    /// The number of iterations for FCI
    int fci_iterations_;
    /// Print the Natural Orbitals from the 1-RDM
    bool print_no_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();
};

}}

#endif // _fci_solver_h_

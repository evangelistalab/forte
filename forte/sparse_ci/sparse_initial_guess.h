#ifndef _sparse_initial_guess_h_
#define _sparse_initial_guess_h_

#include <vector>
// #include "sparse_ci/determinant.h"

namespace forte {

class DavidsonLiuSolver;
class ActiveSpaceIntegrals;

/// @brief Return the Hamiltonian matrix transformed into the S^2 eigenbasis
/// @param guess_dets A vector of guess determinants
/// @param as_ints The active space integrals object
/// @return A tuple of the form (S^2 transformed Hamiltonian matrix, S^2 eigenvalues, S^2
/// eigenvectors)
std::tuple<psi::Matrix, psi::Vector, psi::Matrix>
compute_s2_transformed_hamiltonian_matrix(const std::vector<Determinant>& dets,
                                          std::shared_ptr<ActiveSpaceIntegrals> as_ints);

/// @brief Generate initial guess vectors for the Davidson-Liu solver starting from a set of guess
/// determinants
/// @param guess_dets A vector of guess determinants
/// @param guess_dets_pos A vector of the positions of the guess determinants in the CI vector
/// @param num_guess_states The number of guess states to generate
/// @param as_ints The active space integrals object
/// @param dls The Davidson-Liu solver object
/// @param multiplicity The desired multiplicity of the guess states
/// @param do_spin_project Whether or not to project out guess states with the wrong multiplicity
/// @param user_guess A vector of vectors of pairs of the form (determinant index, coefficient)
/// passed in by the user. If this vector is not empty, then we will use the user guess instead.
/// Spin projection can be still applied.
void find_initial_guess_det(const std::vector<Determinant>& guess_dets,
                            const std::vector<size_t>& guess_dets_pos, size_t num_guess_states,
                            const std::shared_ptr<ActiveSpaceIntegrals>& as_ints,
                            DavidsonLiuSolver& dls, int multiplicity, bool do_spin_project,
                            const std::vector<std::vector<std::pair<size_t, double>>>& user_guess);

} // namespace forte

#endif // _sparse_initial_guess_h_

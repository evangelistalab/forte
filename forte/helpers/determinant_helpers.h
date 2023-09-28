#ifndef _determinant_helpers_h_
#define _determinant_helpers_h_

#include <vector>

#include "sparse_ci/determinant.h"

namespace psi {
class Matrix;
} // namespace psi

namespace forte {

class ActiveSpaceIntegrals;

/// @brief Build the S^2 operator matrix in the given basis of determinants (multithreaded)
/// @param dets A vector of determinants
/// @return A matrix of size (num_dets, num_dets) with the S^2 operator matrix
std::shared_ptr<psi::Matrix> make_s2_matrix(const std::vector<Determinant>& dets);

/// @brief Build the Hamiltonian operator matrix in the given basis of determinants (multithreaded)
/// @param dets A vector of determinants
/// @param as_ints A pointer to the ActiveSpaceIntegrals object
/// @return A matrix of size (num_dets, num_dets) with the Hamiltonian operator matrix
std::shared_ptr<psi::Matrix> make_hamiltonian_matrix(const std::vector<Determinant>& dets,
                                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints);

} // namespace forte

#endif // _determinant_helpers_h_

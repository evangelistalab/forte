#ifndef __LAPLACE_H__
#define __LAPLACE_H__

#include "psi4/libmints/matrix.h"
#include "mrdsrg-spin-integrated/master_mrdsrg.h"


#include <vector>

typedef std::vector<std::vector<int>> SparseMap;

namespace psi {
class PSIO;
} // namespace psi

namespace forte {
/// Written by Shuhang Li

std::vector<int> merge_lists(const std::vector<int> &l1, const std::vector<int> &l2);

/* Args: sorted list of y, sparse map from A to another list of y (assume sorted, each possible value of y appears exactly once in entire map)
    * Returns: the union of lists in A_to_y where at least one element is in y
    */
std::vector<int> contract_lists(const std::vector<int> &y, const std::vector<std::vector<int>> &A_to_y);

/* Args: x is a list of values (sorted), y is a map from values of x to values of y
    * Returns: a list of y values
    *
    * Multiple values in x may map to the same value in y (i.e. x is a list of bf, y is atoms)
    */
std::vector<int> block_list(const std::vector<int> &x_list, const std::vector<int> &x_to_y_map);

/* Args: SparseMap from x to y, maximum possible y value
* Returns: SparseMap from y to x
*/
SparseMap invert_map(const SparseMap &x_to_y, int ny);

/* Args: SparseMap from x to y, SparseMap from y to z
* Returns: SparseMap from x to z
*/
SparseMap chain_maps(const SparseMap &x_to_y, const SparseMap &y_to_z);

/* Args: SparseMap from x to y, list of pairs of type x
* Returns: extended SparseMap from x to y
*/
SparseMap extend_maps(const SparseMap &i_to_y, const std::vector<std::pair<int,int>> &ij_to_i_j);

/* Args: Matrix, list of row indices */
psi::SharedMatrix submatrix_rows(const psi::Matrix &mat, const std::vector<int> &row_inds);

/* Args: Matrix, list of column indices */
psi::SharedMatrix submatrix_cols(const psi::Matrix &mat, const std::vector<int> &col_inds);

/* Args: Matrix, list of row and column indices */
psi::SharedMatrix submatrix_rows_and_cols(const psi::Matrix &mat, const std::vector<int> &row_inds, const std::vector<int> &col_inds);

psi::SharedMatrix load_Amn(const size_t A, const size_t mn);

psi::SharedMatrix load_Jinv_full(const size_t P, const size_t Q);

psi::SharedMatrix initialize_erfc_integral(double Omega, int n_func_pairs, std::shared_ptr<ForteIntegrals> ints_forte);

psi::SharedMatrix erfc_metric (double Omega, std::shared_ptr<ForteIntegrals> ints_forte);

int binary_search_recursive(std::vector<int> A, int key, int low, int high);
} // namespace forte

#endif
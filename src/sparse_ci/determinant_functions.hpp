#ifndef _determinant_functions_hpp_
#define _determinant_functions_hpp_

#include <string>
#include <cstddef>
#include <iostream>
#include <vector>

namespace forte {

#define PERFORMANCE_OPTIMIZATION 0

double slater_rules_single_alpha(String Ib, String Ia, String Ja,
                                 const std::shared_ptr<ActiveSpaceIntegrals>& ints);
double slater_rules_single_beta(String Ia, String Ib, String Jb,
                                const std::shared_ptr<ActiveSpaceIntegrals>& ints);

double slater_rules_double_alpha_alpha(String Ia, String Ja,
                                       const std::shared_ptr<ActiveSpaceIntegrals>& ints);

double slater_rules_double_beta_beta(String Ib, String Jb,
                                     const std::shared_ptr<ActiveSpaceIntegrals>& ints);

double slater_rules_double_alpha_beta_pre(int i, int a, String Ib, String Jb,
                                          const std::shared_ptr<ActiveSpaceIntegrals>& ints);

Determinant common_occupation(const Determinant& lhs, const Determinant& rhs);

Determinant different_occupation(const Determinant& lhs, const Determinant& rhs);

/// Find the spin orbitals that are occupied only one determinant (performs a bitwise OR, |)
Determinant union_occupation(const Determinant& lhs, const Determinant& rhs);

void enforce_spin_completeness(std::vector<Determinant>& det_space, int nmo);

} // namespace forte

#endif // _determinant_functions_hpp_

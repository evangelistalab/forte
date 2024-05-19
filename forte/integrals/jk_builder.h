/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#pragma once

#include <vector>

#include "psi4/libfock/jk.h"
#include "psi4/libmints/dimension.h"
#include "ambit/blocked_tensor.h"

class Tensor;

namespace psi {
class Options;
class Matrix;
class Vector;
class Wavefunction;
class Dimension;
class BasisSet;
} // namespace psi

namespace forte {

class ForteOptions;
class MOSpaceInfo;
class Orbitals;

class JKBuilder {
  public:
    virtual ~JKBuilder() {}

    // Pure virtual function to build the J and K matrices
    virtual void buildJK() = 0;

    // Additional common utility functions can be added here
  protected:
    // Protected constructor to prevent instantiation of base class
    JKBuilder() {}
};

class Psi4JKBuilder : public JKBuilder {
  public:
    Psi4JKBuilder(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> wfn);
    ~Psi4JKBuilder();
    void buildJK() override;

  private:
    std::shared_ptr<ForteOptions> options_;
    std::shared_ptr<psi::Wavefunction> wfn_;
    std::shared_ptr<psi::JK> JK_;
};

class CustomJKBuilder : public JKBuilder {
  public:
    CustomJKBuilder();
    ~CustomJKBuilder();
    void buildJK() override;
};

std::shared_ptr<JKBuilder> make_psi4jk(std::shared_ptr<ForteOptions> options,
                                       std::shared_ptr<psi::Wavefunction> wfn);

} // namespace forte

/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
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

#ifndef _helpers_timer_h_
#define _helpers_timer_h_

#include "psi4/libqt/qt.h"

#include <chrono>

namespace forte {

/**
 * @brief A timer class that returns the elapsed time
 */
class local_timer {
  public:
    local_timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    /// return the elapsed time in seconds
    double get() {
        auto duration = std::chrono::high_resolution_clock::now() - start_;
        return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    }

  private:
    /// stores the time when this object is created
    std::chrono::high_resolution_clock::time_point start_;
};

/**
  * @brief A timer class that prints timing to a file (timer.dat)
  */
class timer {
  public:
    timer(const std::string& name) : name_(name) { psi::timer_on(name_); }
    ~timer() { stop(); }

    /// Return the elapsed time in seconds
    void stop() {
        if (running_) {
            running_ = false;
            psi::timer_off(name_);
        }
    }

  private:
    std::string name_;
    bool running_ = true;
};

/**
  * @brief A timer class
  */
class parallel_timer {
  public:
    parallel_timer(const std::string& name, int rank) : name_(name), rank_(rank) {
        psi::parallel_timer_on(name_, rank_);
    }
    ~parallel_timer() { stop(); }

    /// Return the elapsed time in seconds
    void stop() {
        if (running_) {
            running_ = false;
            psi::parallel_timer_off(name_, rank_);
        }
    }

  private:
    std::string name_;
    int rank_;
    bool running_ = true;
};
}

#endif // _helpers_timer_h_

/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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
 * @brief A timer class to track the elapsed time
 *
 * This class is based on std::chrono::high_resolution_clock and should be used
 * to time functions. The timer is set at creation and the time difference can be
 * obtained with the get() function. The reset() function can be used to reset the
 * timer.
 */
class local_timer {
  public:
    /// constructor. Creates and starts the timer object
    local_timer() : start_(std::chrono::high_resolution_clock::now()) {}

    /// reset the timer
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
 *
 * This class uses the psi4 functions timer_on/timer_off and a local_timer object
 * to track time. The function stop() will return the elapsed time and stop the psi4
 * timer.
 */
class timer {
  public:
    /// constructor. Create a timer with label name
    timer(const std::string& name) : name_(name) {
        psi::timer_on(name_);
        t_ = local_timer();
    }
    ~timer() { stop(); }

    /// Return the elapsed time in seconds
    double stop() {
        if (running_) {
            running_ = false;
            psi::timer_off(name_);
            return t_.get();
        }
        return 0.0;
    }

  private:
    std::string name_;
    bool running_ = true;
    local_timer t_;
};

/**
 * @brief A timer class for parallel functions that prints timing to a file (timer.dat)
 *
 * This class uses the psi4 functions parallel_timer_on/parallel_timer_off to track time.
 * The function stop() will return the elapsed time and stop the psi4 timer.
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
} // namespace forte

#endif // _helpers_timer_h_

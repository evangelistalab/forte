/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <string>
#include <vector>
#include <mutex>

namespace forte {

class Observer {
  public:
    virtual ~Observer() = default;
    virtual void update(const std::string& message) = 0;
};

class Subject {
  public:
    void attach_observer(std::shared_ptr<Observer> observer, const std::string& name) {
        std::scoped_lock lock(mutex_);
        observers_.emplace_back(name, observer);
    }

  protected:
    void notify_observers(const std::string& message = std::string()) {
        std::scoped_lock lock(mutex_);
        for (auto& [name, observer] : observers_) {
            // if the observer is still alive, notify it
            if (auto obs = observer.lock()) {
                obs->update(message);
            }
        }
    }

  private:
    std::vector<std::pair<std::string, std::weak_ptr<Observer>>> observers_;
    std::mutex mutex_;
};

} // namespace forte

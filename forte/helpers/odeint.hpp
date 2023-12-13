#pragma once

#include <vector>

using odeint_state_type = std::vector<double>;

using ODEFunction =
    std::function<void(const odeint_state_type& x, odeint_state_type& dxdt, const double t)>;

using ODECallback = std::function<void(const odeint_state_type& x, const double t)>;

void runge_kutta_4_step(const ODEFunction& f, double t, const odeint_state_type& x,
                        odeint_state_type& x_next, odeint_state_type& x_temp, odeint_state_type& k,
                        double h) {
    size_t maxi = x.size();

    for (size_t i = 0; i < maxi; ++i) {
        x_next[i] = x[i];
        x_temp[i] = x[i];
    }

    // step 1
    f(x, k, t);
    for (size_t i = 0; i < maxi; ++i) {
        x_temp[i] = x[i] + 0.5 * h * k[i];
    }
    for (size_t i = 0; i < maxi; ++i) {
        x_next[i] += h * k[i] / 6.0;
    }

    // step 2
    f(x_temp, k, t + 0.5 * h);
    for (size_t i = 0; i < maxi; ++i) {
        x_temp[i] = x[i] + 0.5 * h * k[i];
    }
    for (size_t i = 0; i < maxi; ++i) {
        x_next[i] += h * k[i] / 3.0;
    }

    f(x_temp, k, t + 0.5 * h);
    for (size_t i = 0; i < maxi; ++i) {
        x_temp[i] = x[i] + h * k[i];
    }
    for (size_t i = 0; i < maxi; ++i) {
        x_next[i] += h * k[i] / 3.0;
    }

    f(x_temp, k, t + h);
    for (size_t i = 0; i < maxi; ++i) {
        x_next[i] += h * k[i] / 6.0;
    }
}

void runge_kutta_4_adaptive(const ODEFunction& f, const ODECallback& callback, odeint_state_type& x,
                            double t_init, double t_end, double h, double tolerance) {
    size_t n = x.size();
    odeint_state_type x_temp(n), x1(n), x2(n), x3(n), k(n);
    callback(x, t_init);
    for (double t = t_init; t < t_end;) {
        if (t + h > t_end) {
            h = t_end - t;
        }
        runge_kutta_4_step(f, t, x, x1, x_temp, k, h);
        runge_kutta_4_step(f, t, x, x2, x_temp, k, h / 2.0);
        runge_kutta_4_step(f, t, x2, x3, x_temp, k, h / 2.0);

        double max_error = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double error = std::abs(x1[i] - x3[i]);
            max_error = std::max(max_error, error);
        }

        if (max_error < tolerance) {
            // Accept step and update t and y
            x = x1;
            t += h;
            if (max_error < tolerance / 4.0) {
                h *= 2.0;
            }
            callback(x, t);
        } else {
            // Reject step and try again
            h /= 2.0;
        }
    }
}
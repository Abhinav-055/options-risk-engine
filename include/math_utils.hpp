#pragma once

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <limits>

namespace quant::math {

[[nodiscard]] inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * std::numbers::sqrt2 / 2.0);
}

[[nodiscard]] inline double norm_pdf(double x) noexcept {
    static const double inv_sqrt_2pi =
        1.0 / (std::numbers::sqrt2 * std::sqrt(std::numbers::pi));
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

[[nodiscard]] inline double log_sum_exp(double a, double b) noexcept {
    const double m = std::max(a, b);
    if (m == -std::numeric_limits<double>::infinity()) {
        return -std::numeric_limits<double>::infinity();
    }
    return m + std::log1p(std::exp(-std::abs(a - b)));
}

}

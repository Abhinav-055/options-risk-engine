#pragma once

#include "math_utils.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <limits>

namespace quant {

enum class OptionType { Call, Put };

struct BSParams {
    double S;
    double K;
    double r;
    double q;
    double sigma;
    double T;
};

struct D1D2 {
    double d1;
    double d2;
};

inline void validate_params(const BSParams& p) {
    if (p.S <= 0.0) throw std::invalid_argument("Spot price S must be positive");
    if (p.K <= 0.0) throw std::invalid_argument("Strike price K must be positive");
    if (p.sigma < 0.0) throw std::invalid_argument("Volatility sigma must be non-negative");
    if (p.T < 0.0) throw std::invalid_argument("Time to expiry T must be non-negative");
}

constexpr double BS_EPSILON = 1e-14;

[[nodiscard]] inline D1D2 compute_d1d2(const BSParams& p) noexcept {
    const double sqrt_T = std::sqrt(p.T);
    const double vol_sqrt_T = p.sigma * sqrt_T;
    const double d1 = (std::log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma * p.sigma) * p.T)
                       / vol_sqrt_T;
    const double d2 = d1 - vol_sqrt_T;
    return {d1, d2};
}

[[nodiscard]] double call_price(const BSParams& p);

[[nodiscard]] double put_price(const BSParams& p);

[[nodiscard]] double bs_price(const BSParams& p, OptionType type);

}

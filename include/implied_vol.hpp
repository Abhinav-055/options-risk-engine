#pragma once

#include "black_scholes.hpp"
#include "greeks.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace quant {

struct IVResult {
    double iv;
    int nr_iterations;
    bool used_bisection;
    double residual;
};

struct IVConfig {
    int max_nr_iterations = 100;
    int max_bisect_iterations = 200;
    double tolerance = 1e-8;
    double initial_guess = 0.25;
    double bisect_lower = 1e-6;
    double bisect_upper = 10.0;
    double vega_floor = 1e-12;
};

[[nodiscard]] IVResult solve_implied_vol(
    double market_price,
    double S, double K, double r, double q, double T,
    OptionType type,
    const IVConfig& config = IVConfig{});

}

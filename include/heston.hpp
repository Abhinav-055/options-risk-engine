#pragma once

#include "math_utils.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <future>
#include <chrono>
#include <algorithm>
#include <numbers>
#include <cstdint>

namespace quant {

struct HestonParams {
    double S0;
    double K;
    double r;
    double q;
    double v0;
    double kappa;
    double theta;
    double xi;
    double rho;
    double T;
};

struct HestonResult {
    double price;
    double standard_error;
    double ci_lower;
    double ci_upper;
    std::size_t paths_used;
    double elapsed_ms;
    bool feller_satisfied;
};

[[nodiscard]] HestonResult heston_mc_price(
    const HestonParams& params,
    std::size_t num_paths = 500'000,
    std::size_t num_steps = 252,
    unsigned int seed = 42);

}

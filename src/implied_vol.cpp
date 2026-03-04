
#include "implied_vol.hpp"
#include "black_scholes.hpp"
#include "greeks.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace quant {

[[nodiscard]] IVResult solve_implied_vol(
    double market_price,
    double S, double K, double r, double q, double T,
    OptionType type,
    const IVConfig& config)
{
    if (market_price <= 0.0) {
        throw std::invalid_argument("Market price must be positive");
    }

    auto price_at = [&](double sigma) -> double {
        BSParams params{S, K, r, q, sigma, T};
        return bs_price(params, type);
    };

    auto vega_at = [&](double sigma) -> double {
        BSParams params{S, K, r, q, sigma, T};
        return compute_greeks(params, type).vega;
    };

    double sigma = config.initial_guess;
    int nr_iters = 0;
    bool nr_converged = false;

    for (int i = 0; i < config.max_nr_iterations; ++i) {
        nr_iters = i + 1;

        const double bs_val = price_at(sigma);
        const double diff = bs_val - market_price;

        if (std::abs(diff) < config.tolerance) {
            nr_converged = true;
            return IVResult{sigma, nr_iters, false, std::abs(diff)};
        }

        const double v = vega_at(sigma);

        if (std::abs(v) < config.vega_floor) {
            break;
        }

        double sigma_new = sigma - diff / v;

        if (sigma_new <= 0.0) {
            sigma_new = sigma * 0.5;
        }

        sigma_new = std::min(sigma_new, config.bisect_upper);

        sigma = sigma_new;
    }

    double lo = config.bisect_lower;
    double hi = config.bisect_upper;

    double f_lo = price_at(lo) - market_price;
    double f_hi = price_at(hi) - market_price;

    if (f_lo * f_hi > 0.0) {
        throw std::runtime_error(
            "Implied volatility solver: no root in bracket [" +
            std::to_string(lo) + ", " + std::to_string(hi) +
            "]. f(lo)=" + std::to_string(f_lo) +
            ", f(hi)=" + std::to_string(f_hi) +
            ". Market price may be outside feasible range.");
    }

    for (int i = 0; i < config.max_bisect_iterations; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double f_mid = price_at(mid) - market_price;

        if (std::abs(f_mid) < config.tolerance || (hi - lo) < config.tolerance) {
            return IVResult{mid, nr_iters, true, std::abs(f_mid)};
        }

        if (f_mid * f_lo < 0.0) {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }

    const double final_sigma = 0.5 * (lo + hi);
    return IVResult{final_sigma, nr_iters, true, std::abs(price_at(final_sigma) - market_price)};
}

}

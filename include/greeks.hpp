#pragma once

#include "black_scholes.hpp"
#include "math_utils.hpp"
#include <cmath>
#include <numbers>

namespace quant {

struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
    double vanna;
    double volga;
};

[[nodiscard]] inline Greeks compute_greeks(const BSParams& p, OptionType type) {
    validate_params(p);

    Greeks g{};

    if (p.T < BS_EPSILON || p.sigma < BS_EPSILON) {
        if (type == OptionType::Call) {
            g.delta = (p.S > p.K) ? 1.0 : 0.0;
        } else {
            g.delta = (p.S < p.K) ? -1.0 : 0.0;
        }
        return g;
    }

    const double sqrt_T = std::sqrt(p.T);
    const double vol_sqrt_T = p.sigma * sqrt_T;
    const auto [d1, d2] = compute_d1d2(p);
    const double nd1 = math::norm_pdf(d1);
    const double Nd1 = math::norm_cdf(d1);
    const double Nd2 = math::norm_cdf(d2);
    const double div_disc = std::exp(-p.q * p.T);
    const double disc = std::exp(-p.r * p.T);

    if (type == OptionType::Call) {
        g.delta = div_disc * Nd1;
    } else {
        g.delta = -div_disc * math::norm_cdf(-d1);
    }

    g.gamma = div_disc * nd1 / (p.S * vol_sqrt_T);

    g.vega = p.S * div_disc * nd1 * sqrt_T;

    const double common_theta = -p.S * div_disc * nd1 * p.sigma / (2.0 * sqrt_T);
    if (type == OptionType::Call) {
        g.theta = common_theta
                  + p.q * p.S * div_disc * Nd1
                  - p.r * p.K * disc * Nd2;
    } else {
        g.theta = common_theta
                  - p.q * p.S * div_disc * math::norm_cdf(-d1)
                  + p.r * p.K * disc * math::norm_cdf(-d2);
    }

    if (type == OptionType::Call) {
        g.rho = p.K * p.T * disc * Nd2;
    } else {
        g.rho = -p.K * p.T * disc * math::norm_cdf(-d2);
    }

    g.vanna = -div_disc * nd1 * d2 / p.sigma;

    g.volga = g.vega * d1 * d2 / p.sigma;

    return g;
}

}

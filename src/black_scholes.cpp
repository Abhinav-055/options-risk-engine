
#include "black_scholes.hpp"
#include "math_utils.hpp"
#include <cmath>
#include <algorithm>

namespace quant {

[[nodiscard]] double call_price(const BSParams& p) {
    validate_params(p);

    if (p.T < BS_EPSILON) {
        return std::max(p.S - p.K, 0.0);
    }

    if (p.sigma < BS_EPSILON) {
        const double forward = p.S * std::exp((p.r - p.q) * p.T);
        if (forward > p.K) {
            return p.S * std::exp(-p.q * p.T) - p.K * std::exp(-p.r * p.T);
        }
        return 0.0;
    }

    const auto [d1, d2] = compute_d1d2(p);
    const double discount = std::exp(-p.r * p.T);
    const double div_discount = std::exp(-p.q * p.T);

    return p.S * div_discount * math::norm_cdf(d1) - p.K * discount * math::norm_cdf(d2);
}

[[nodiscard]] double put_price(const BSParams& p) {
    validate_params(p);

    if (p.T < BS_EPSILON) {
        return std::max(p.K - p.S, 0.0);
    }

    if (p.sigma < BS_EPSILON) {
        const double forward = p.S * std::exp((p.r - p.q) * p.T);
        if (forward < p.K) {
            return p.K * std::exp(-p.r * p.T) - p.S * std::exp(-p.q * p.T);
        }
        return 0.0;
    }

    const auto [d1, d2] = compute_d1d2(p);
    const double discount = std::exp(-p.r * p.T);
    const double div_discount = std::exp(-p.q * p.T);

    return p.K * discount * math::norm_cdf(-d2) - p.S * div_discount * math::norm_cdf(-d1);
}

[[nodiscard]] double bs_price(const BSParams& p, OptionType type) {
    return (type == OptionType::Call) ? call_price(p) : put_price(p);
}

}

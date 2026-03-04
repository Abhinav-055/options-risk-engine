
#include "binomial.hpp"
#include "black_scholes.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace quant {

[[nodiscard]] BinomialResult binomial_american(
    double S, double K, double r, double q, double sigma, double T,
    OptionType type,
    int steps)
{
    if (S <= 0.0) throw std::invalid_argument("Spot price S must be positive");
    if (K <= 0.0) throw std::invalid_argument("Strike price K must be positive");
    if (sigma <= 0.0) throw std::invalid_argument("Volatility sigma must be positive for binomial tree");
    if (T <= 0.0) throw std::invalid_argument("Time to expiry T must be positive for binomial tree");
    if (steps < 1) throw std::invalid_argument("Number of steps must be >= 1");

    const double dt = T / static_cast<double>(steps);
    const double sqrt_dt = std::sqrt(dt);

    const double u = std::exp(sigma * sqrt_dt);
    const double d = 1.0 / u;
    const double growth = std::exp((r - q) * dt);
    const double p_up = (growth - d) / (u - d);
    const double p_down = 1.0 - p_up;
    const double disc = std::exp(-r * dt);

    if (p_up <= 0.0 || p_up >= 1.0) {
        throw std::runtime_error(
            "Risk-neutral probability out of bounds: p=" + std::to_string(p_up) +
            ". Try increasing the number of steps or checking parameters.");
    }

    const int n = steps;
    std::vector<double> V(n + 1);

    for (int j = 0; j <= n; ++j) {
        const double S_node = S * std::pow(u, 2 * j - n);
        if (type == OptionType::Call) {
            V[j] = std::max(S_node - K, 0.0);
        } else {
            V[j] = std::max(K - S_node, 0.0);
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            const double continuation = disc * (p_up * V[j + 1] + p_down * V[j]);

            const double S_node = S * std::pow(u, 2 * j - i);

            double exercise;
            if (type == OptionType::Call) {
                exercise = std::max(S_node - K, 0.0);
            } else {
                exercise = std::max(K - S_node, 0.0);
            }

            V[j] = std::max(continuation, exercise);
        }
    }

    return BinomialResult{V[0], steps};
}

}

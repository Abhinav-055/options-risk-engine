
#include "heston.hpp"
#include "math_utils.hpp"
#include <cmath>
#include <random>
#include <thread>
#include <future>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numbers>

namespace quant {

static std::pair<double, double> heston_simulate_thread(
    const HestonParams& p,
    std::size_t num_paths,
    std::size_t num_steps,
    std::uint64_t thread_seed)
{
    std::seed_seq sseq{thread_seed, thread_seed >> 32, thread_seed * 2862933555777941757ULL};
    std::mt19937_64 rng(sseq);
    std::normal_distribution<double> norm(0.0, 1.0);

    const double dt = p.T / static_cast<double>(num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double discount = std::exp(-p.r * p.T);

    const double rho_comp = std::sqrt(std::max(1.0 - p.rho * p.rho, 0.0));

    double sum = 0.0;
    double sum_sq = 0.0;

    for (std::size_t i = 0; i < num_paths; ++i) {
        double S = p.S0;
        double v = p.v0;

        for (std::size_t step = 0; step < num_steps; ++step) {
            const double W1 = norm(rng);
            const double W2 = norm(rng);

            const double Z1 = W1;
            const double Z2 = p.rho * W1 + rho_comp * W2;

            const double v_plus = std::max(v, 0.0);
            const double sqrt_v = std::sqrt(v_plus);

            S = S * std::exp((p.r - p.q - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * Z1);

            v = v + p.kappa * (p.theta - v_plus) * dt + p.xi * sqrt_v * sqrt_dt * Z2;

            v = std::max(v, 0.0);
        }

        const double payoff = discount * std::max(S - p.K, 0.0);
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    return {sum, sum_sq};
}

[[nodiscard]] HestonResult heston_mc_price(
    const HestonParams& params,
    std::size_t num_paths,
    std::size_t num_steps,
    unsigned int seed)
{
    const auto t_start = std::chrono::high_resolution_clock::now();

    const bool feller = (2.0 * params.kappa * params.theta > params.xi * params.xi);

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    const unsigned int nthreads = (hw_threads > 0) ? hw_threads : 4;

    const std::size_t base = num_paths / nthreads;
    const std::size_t remainder = num_paths % nthreads;

    std::vector<std::future<std::pair<double, double>>> futures;
    futures.reserve(nthreads);

    for (unsigned int t = 0; t < nthreads; ++t) {
        const std::size_t paths_this = base + (t < remainder ? 1 : 0);
        const std::uint64_t tseed = static_cast<std::uint64_t>(seed) * 6364136223846793005ULL + t;

        futures.push_back(std::async(std::launch::async,
            heston_simulate_thread, std::cref(params), paths_this, num_steps, tseed));
    }

    double total_sum = 0.0;
    double total_sum_sq = 0.0;
    std::size_t total_count = 0;

    for (unsigned int t = 0; t < nthreads; ++t) {
        auto [s, ssq] = futures[t].get();
        total_sum += s;
        total_sum_sq += ssq;
        total_count += base + (t < remainder ? 1 : 0);
    }

    const double n = static_cast<double>(total_count);
    const double mean = total_sum / n;
    const double var = (total_sum_sq / n) - mean * mean;
    const double se = std::sqrt(std::max(var, 0.0) / n);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return HestonResult{
        mean, se,
        mean - 1.96 * se, mean + 1.96 * se,
        total_count, elapsed, feller
    };
}

}

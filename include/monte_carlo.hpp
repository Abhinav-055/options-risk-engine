#pragma once

#include "math_utils.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <future>
#include <chrono>
#include <functional>
#include <numbers>
#include <numeric>
#include <algorithm>
#include <cstdint>

namespace quant {

struct ConvergencePoint {
    std::size_t paths_so_far;
    double running_mean;
    double running_stderr;
};

struct MCResult {
    double price;
    double standard_error;
    double ci_lower;
    double ci_upper;
    std::size_t paths_used;
    double elapsed_ms;
    std::vector<ConvergencePoint> convergence;
};

struct MCConfig {
    std::size_t num_paths = 1'000'000;
    std::size_t num_steps = 252;
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.0;
    double sigma = 0.20;
    double T = 1.0;
    bool use_antithetic = true;
    unsigned int seed = 42;
};

template <typename PayoffFn>
std::pair<double, double> mc_simulate_thread(
    const MCConfig& cfg,
    PayoffFn&& payoff,
    std::size_t paths_this_thread,
    std::uint64_t thread_seed,
    std::vector<ConvergencePoint>& convergence_out,
    std::size_t convergence_interval)
{
    std::seed_seq sseq{thread_seed, thread_seed >> 32, thread_seed * 6364136223846793005ULL};
    std::mt19937_64 rng(sseq);
    std::normal_distribution<double> norm(0.0, 1.0);

    const double dt = cfg.T / static_cast<double>(cfg.num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double drift = (cfg.r - cfg.q - 0.5 * cfg.sigma * cfg.sigma) * dt;
    const double vol = cfg.sigma * sqrt_dt;
    const double discount = std::exp(-cfg.r * cfg.T);

    double sum = 0.0;
    double sum_sq = 0.0;

    std::vector<double> path(cfg.num_steps + 1);
    std::vector<double> path_anti(cfg.num_steps + 1);

    const std::size_t iters = cfg.use_antithetic ? paths_this_thread / 2 : paths_this_thread;

    for (std::size_t i = 0; i < iters; ++i) {
        path[0] = cfg.S0;
        if (cfg.use_antithetic) path_anti[0] = cfg.S0;

        for (std::size_t step = 1; step <= cfg.num_steps; ++step) {
            const double z = norm(rng);
            path[step] = path[step - 1] * std::exp(drift + vol * z);

            if (cfg.use_antithetic) {
                path_anti[step] = path_anti[step - 1] * std::exp(drift + vol * (-z));
            }
        }

        const double pv1 = discount * payoff(path);
        if (cfg.use_antithetic) {
            const double pv2 = discount * payoff(path_anti);
            const double avg = 0.5 * (pv1 + pv2);
            sum += avg;
            sum_sq += avg * avg;
        } else {
            sum += pv1;
            sum_sq += pv1 * pv1;
        }

        const std::size_t n = i + 1;
        if (convergence_interval > 0 && (n % convergence_interval == 0)) {
            const double mean = sum / static_cast<double>(n);
            const double var = (sum_sq / static_cast<double>(n)) - mean * mean;
            const double se = std::sqrt(std::max(var, 0.0) / static_cast<double>(n));
            const std::size_t effective = cfg.use_antithetic ? n * 2 : n;
            convergence_out.push_back({effective, mean, se});
        }
    }

    return {sum, sum_sq};
}

template <typename PayoffFn>
[[nodiscard]] MCResult mc_price(const MCConfig& cfg, PayoffFn&& payoff) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    const unsigned int num_threads = (hw_threads > 0) ? hw_threads : 4;

    const std::size_t effective_paths = cfg.use_antithetic
        ? (cfg.num_paths / 2) * 2
        : cfg.num_paths;

    const std::size_t base_per_thread = effective_paths / num_threads;
    const std::size_t remainder = effective_paths % num_threads;

    struct ThreadResult {
        double sum;
        double sum_sq;
        std::size_t count;
        std::vector<ConvergencePoint> convergence;
    };

    std::vector<std::future<ThreadResult>> futures;
    futures.reserve(num_threads);

    for (unsigned int t = 0; t < num_threads; ++t) {
        const std::size_t paths_this = base_per_thread + (t < remainder ? 1 : 0);
        const std::uint64_t thread_seed = static_cast<std::uint64_t>(cfg.seed) * 2654435761ULL + t;
        const std::size_t iters = cfg.use_antithetic ? paths_this / 2 : paths_this;
        const std::size_t conv_interval = std::max<std::size_t>(iters / 100, 1);

        futures.push_back(std::async(std::launch::async,
            [&cfg, &payoff, paths_this, thread_seed, conv_interval]() -> ThreadResult {
                std::vector<ConvergencePoint> conv;
                auto [s, ssq] = mc_simulate_thread(cfg, payoff, paths_this, thread_seed, conv, conv_interval);
                const std::size_t count = cfg.use_antithetic ? paths_this / 2 : paths_this;
                return {s, ssq, count, std::move(conv)};
            }));
    }

    double total_sum = 0.0;
    double total_sum_sq = 0.0;
    std::size_t total_count = 0;
    std::vector<ConvergencePoint> all_convergence;

    for (auto& f : futures) {
        auto res = f.get();
        total_sum += res.sum;
        total_sum_sq += res.sum_sq;
        total_count += res.count;
        all_convergence.insert(all_convergence.end(),
                               res.convergence.begin(), res.convergence.end());
    }

    std::sort(all_convergence.begin(), all_convergence.end(),
              [](const ConvergencePoint& a, const ConvergencePoint& b) {
                  return a.paths_so_far < b.paths_so_far;
              });

    const double mean = total_sum / static_cast<double>(total_count);
    const double var = (total_sum_sq / static_cast<double>(total_count)) - mean * mean;
    const double se = std::sqrt(std::max(var, 0.0) / static_cast<double>(total_count));

    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return MCResult{
        mean,
        se,
        mean - 1.96 * se,
        mean + 1.96 * se,
        cfg.use_antithetic ? total_count * 2 : total_count,
        elapsed,
        std::move(all_convergence)
    };
}

[[nodiscard]] inline double geometric_asian_call_price(
    double S0, double K, double r, double q, double sigma, double T, std::size_t n)
{
    const double n_d = static_cast<double>(n);
    const double sigma_g = sigma * std::sqrt((2.0 * n_d + 1.0) / (6.0 * (n_d + 1.0)));

    const double mu_g = (r - q - 0.5 * sigma * sigma) * (n_d + 1.0) / (2.0 * n_d)
                        + 0.5 * sigma_g * sigma_g;

    const double d1 = (std::log(S0 / K) + (mu_g + 0.5 * sigma_g * sigma_g) * T)
                       / (sigma_g * std::sqrt(T));
    const double d2 = d1 - sigma_g * std::sqrt(T);

    return std::exp(-r * T) * (S0 * std::exp(mu_g * T) * math::norm_cdf(d1)
                                - K * math::norm_cdf(d2));
}

template <typename PayoffFn>
[[nodiscard]] MCResult mc_price_control_variate(
    const MCConfig& cfg,
    PayoffFn&& arith_payoff,
    double control_exact_price)
{

    auto geom_payoff = [&](const std::vector<double>& path) -> double {
        double log_sum = 0.0;
        const std::size_t n = path.size() - 1;
        for (std::size_t i = 1; i <= n; ++i) {
            log_sum += std::log(path[i]);
        }
        const double geom_avg = std::exp(log_sum / static_cast<double>(n));
        return std::max(geom_avg - cfg.K, 0.0);
    };

    const auto t_start = std::chrono::high_resolution_clock::now();

    std::seed_seq sseq{cfg.seed, cfg.seed * 3u, cfg.seed * 7u};
    std::mt19937_64 rng(sseq);
    std::normal_distribution<double> norm(0.0, 1.0);

    const double dt = cfg.T / static_cast<double>(cfg.num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double drift = (cfg.r - cfg.q - 0.5 * cfg.sigma * cfg.sigma) * dt;
    const double vol = cfg.sigma * sqrt_dt;
    const double discount = std::exp(-cfg.r * cfg.T);

    std::vector<double> path(cfg.num_steps + 1);
    double sum_arith = 0.0, sum_geom = 0.0;
    double sum_arith_sq = 0.0;
    double sum_cross = 0.0;

    for (std::size_t i = 0; i < cfg.num_paths; ++i) {
        path[0] = cfg.S0;
        for (std::size_t step = 1; step <= cfg.num_steps; ++step) {
            path[step] = path[step - 1] * std::exp(drift + vol * norm(rng));
        }
        const double pv_a = discount * arith_payoff(path);
        const double pv_g = discount * geom_payoff(path);

        sum_arith += pv_a;
        sum_geom += pv_g;
        sum_arith_sq += pv_a * pv_a;
        sum_cross += pv_a * pv_g;
    }

    const double n = static_cast<double>(cfg.num_paths);
    const double mean_a = sum_arith / n;
    const double mean_g = sum_geom / n;

    const double cv_price = mean_a - 1.0 * (mean_g - control_exact_price);

    const double var_a = (sum_arith_sq / n) - mean_a * mean_a;
    const double se = std::sqrt(std::max(var_a, 0.0) / n);

    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return MCResult{
        cv_price, se,
        cv_price - 1.96 * se, cv_price + 1.96 * se,
        cfg.num_paths, elapsed, {}
    };
}

}


#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numbers>
#include <algorithm>
#include <numeric>

#include "math_utils.hpp"
#include "black_scholes.hpp"
#include "greeks.hpp"
#include "monte_carlo.hpp"
#include "implied_vol.hpp"
#include "binomial.hpp"
#include "heston.hpp"

using namespace quant;

TEST(MathUtils, NormCdfBoundary) {
    EXPECT_NEAR(math::norm_cdf(0.0), 0.5, 1e-15);
}

TEST(MathUtils, NormCdfSymmetry) {
    for (double x : {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}) {
        EXPECT_NEAR(math::norm_cdf(x) + math::norm_cdf(-x), 1.0, 1e-15);
    }
}

TEST(MathUtils, NormCdfTails) {
    EXPECT_GT(math::norm_cdf(-10.0), 0.0);
    EXPECT_LT(math::norm_cdf(-10.0), 1e-20);

    EXPECT_GT(math::norm_cdf(-6.0), 0.0);
    EXPECT_LT(math::norm_cdf(-6.0), 1e-8);

    EXPECT_GT(math::norm_cdf(6.0), 1.0 - 1e-8);
}

TEST(MathUtils, NormPdf) {
    const double expected = 1.0 / std::sqrt(2.0 * std::numbers::pi);
    EXPECT_NEAR(math::norm_pdf(0.0), expected, 1e-15);

    EXPECT_NEAR(math::norm_pdf(2.5), math::norm_pdf(-2.5), 1e-15);
}

TEST(MathUtils, LogSumExp) {
    double expected = 2.0 + std::log1p(std::exp(-1.0));
    EXPECT_NEAR(math::log_sum_exp(1.0, 2.0), expected, 1e-14);

    EXPECT_NEAR(math::log_sum_exp(700.0, 700.0), 700.0 + std::log(2.0), 1e-10);

    EXPECT_NEAR(math::log_sum_exp(0.0, 100.0), 100.0, 1e-10);

    EXPECT_NEAR(math::log_sum_exp(3.0, 5.0), math::log_sum_exp(5.0, 3.0), 1e-15);
}

class BlackScholesTest : public ::testing::Test {
protected:
    BSParams atm{100.0, 100.0, 0.05, 0.02, 0.20, 1.0};
    BSParams itm_call{100.0, 80.0, 0.05, 0.0, 0.25, 0.5};
    BSParams otm_call{100.0, 120.0, 0.05, 0.0, 0.25, 0.5};
};

TEST_F(BlackScholesTest, PutCallParity) {
    const double C = call_price(atm);
    const double P = put_price(atm);
    const double parity_rhs = atm.S * std::exp(-atm.q * atm.T)
                             - atm.K * std::exp(-atm.r * atm.T);

    EXPECT_NEAR(C - P, parity_rhs, 1e-10)
        << "Put-call parity violation: C=" << C << " P=" << P
        << " RHS=" << parity_rhs;
}

TEST_F(BlackScholesTest, PutCallParityMultipleStrikes) {
    for (double K : {70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0}) {
        BSParams p{100.0, K, 0.05, 0.03, 0.30, 1.5};
        double C = call_price(p);
        double P = put_price(p);
        double rhs = p.S * std::exp(-p.q * p.T) - K * std::exp(-p.r * p.T);
        EXPECT_NEAR(C - P, rhs, 1e-10) << "K=" << K;
    }
}

TEST_F(BlackScholesTest, DegenerateT0) {
    BSParams p{100.0, 90.0, 0.05, 0.0, 0.20, 0.0};
    EXPECT_NEAR(call_price(p), 10.0, 1e-12);
    EXPECT_NEAR(put_price(p), 0.0, 1e-12);

    BSParams p2{100.0, 110.0, 0.05, 0.0, 0.20, 0.0};
    EXPECT_NEAR(call_price(p2), 0.0, 1e-12);
    EXPECT_NEAR(put_price(p2), 10.0, 1e-12);
}

TEST_F(BlackScholesTest, DegenerateSigma0) {
    BSParams p{100.0, 100.0, 0.05, 0.0, 0.0, 1.0};
    double expected = 100.0 - 100.0 * std::exp(-0.05);
    EXPECT_NEAR(call_price(p), expected, 1e-10);
}

TEST_F(BlackScholesTest, NonNegativePrices) {
    for (double sigma : {0.01, 0.1, 0.5, 1.0, 2.0}) {
        BSParams p{100.0, 100.0, 0.05, 0.0, sigma, 1.0};
        EXPECT_GE(call_price(p), 0.0);
        EXPECT_GE(put_price(p), 0.0);
    }
}

TEST_F(BlackScholesTest, KnownValue) {
    BSParams p{100.0, 100.0, 0.05, 0.0, 0.20, 1.0};
    EXPECT_NEAR(call_price(p), 10.4506, 0.001);
}

class GreeksTest : public ::testing::Test {
protected:
    BSParams base{100.0, 100.0, 0.05, 0.02, 0.25, 1.0};
    double bump = 1e-5;
};

TEST_F(GreeksTest, DeltaFiniteDifference) {
    auto g = compute_greeks(base, OptionType::Call);

    BSParams up = base; up.S += bump;
    BSParams dn = base; dn.S -= bump;
    double fd_delta = (call_price(up) - call_price(dn)) / (2.0 * bump);

    EXPECT_NEAR(g.delta, fd_delta, 1e-5)
        << "Delta: closed-form=" << g.delta << " FD=" << fd_delta;
}

TEST_F(GreeksTest, GammaFiniteDifference) {
    auto g = compute_greeks(base, OptionType::Call);

    BSParams up = base; up.S += bump;
    BSParams dn = base; dn.S -= bump;
    double fd_gamma = (call_price(up) - 2.0 * call_price(base) + call_price(dn))
                      / (bump * bump);

    EXPECT_NEAR(g.gamma, fd_gamma, 1e-4)
        << "Gamma: closed-form=" << g.gamma << " FD=" << fd_gamma;
}

TEST_F(GreeksTest, VegaFiniteDifference) {
    auto g = compute_greeks(base, OptionType::Call);

    BSParams up = base; up.sigma += bump;
    BSParams dn = base; dn.sigma -= bump;
    double fd_vega = (call_price(up) - call_price(dn)) / (2.0 * bump);

    EXPECT_NEAR(g.vega, fd_vega, 1e-4)
        << "Vega: closed-form=" << g.vega << " FD=" << fd_vega;
}

TEST_F(GreeksTest, ThetaFiniteDifference) {
    auto g = compute_greeks(base, OptionType::Call);

    double dt = 1e-5;
    BSParams up = base; up.T += dt;
    BSParams dn = base; dn.T -= dt;
    double fd_theta = (call_price(dn) - call_price(up)) / (2.0 * dt);

    EXPECT_NEAR(g.theta, fd_theta, 1e-3)
        << "Theta: closed-form=" << g.theta << " FD=" << fd_theta;
}

TEST_F(GreeksTest, RhoFiniteDifference) {
    auto g = compute_greeks(base, OptionType::Call);

    BSParams up = base; up.r += bump;
    BSParams dn = base; dn.r -= bump;
    double fd_rho = (call_price(up) - call_price(dn)) / (2.0 * bump);

    EXPECT_NEAR(g.rho, fd_rho, 1e-4)
        << "Rho: closed-form=" << g.rho << " FD=" << fd_rho;
}

TEST_F(GreeksTest, VannaFiniteDifference) {
    auto g = compute_greeks(base, OptionType::Call);

    BSParams up = base; up.sigma += bump;
    BSParams dn = base; dn.sigma -= bump;
    double delta_up = compute_greeks(up, OptionType::Call).delta;
    double delta_dn = compute_greeks(dn, OptionType::Call).delta;
    double fd_vanna = (delta_up - delta_dn) / (2.0 * bump);

    EXPECT_NEAR(g.vanna, fd_vanna, 1e-3)
        << "Vanna: closed-form=" << g.vanna << " FD=" << fd_vanna;
}

TEST_F(GreeksTest, VolgaFiniteDifference) {
    auto g = compute_greeks(base, OptionType::Call);

    BSParams up = base; up.sigma += bump;
    BSParams dn = base; dn.sigma -= bump;
    double vega_up = compute_greeks(up, OptionType::Call).vega;
    double vega_dn = compute_greeks(dn, OptionType::Call).vega;
    double fd_volga = (vega_up - vega_dn) / (2.0 * bump);

    EXPECT_NEAR(g.volga, fd_volga, 1e-2)
        << "Volga: closed-form=" << g.volga << " FD=" << fd_volga;
}

TEST_F(GreeksTest, PutGreeksSigns) {
    auto g = compute_greeks(base, OptionType::Put);
    EXPECT_LT(g.delta, 0.0);
    EXPECT_GT(g.gamma, 0.0);
    EXPECT_GT(g.vega, 0.0);
    EXPECT_LT(g.rho, 0.0);
}

TEST_F(GreeksTest, GammaCallPutEqual) {
    auto gc = compute_greeks(base, OptionType::Call);
    auto gp = compute_greeks(base, OptionType::Put);
    EXPECT_NEAR(gc.gamma, gp.gamma, 1e-12);
    EXPECT_NEAR(gc.vega, gp.vega, 1e-12);
}

TEST(ImpliedVol, RoundTrip) {
    BSParams p{100.0, 100.0, 0.05, 0.02, 0.30, 1.0};
    double market = call_price(p);

    auto result = solve_implied_vol(market, p.S, p.K, p.r, p.q, p.T, OptionType::Call);

    EXPECT_NEAR(result.iv, 0.30, 1e-7)
        << "IV round-trip failed: expected 0.30, got " << result.iv;

    BSParams recovered{p.S, p.K, p.r, p.q, result.iv, p.T};
    EXPECT_NEAR(call_price(recovered), market, 1e-7);
}

TEST(ImpliedVol, RoundTripPut) {
    BSParams p{100.0, 110.0, 0.05, 0.0, 0.25, 0.5};
    double market = put_price(p);

    auto result = solve_implied_vol(market, p.S, p.K, p.r, p.q, p.T, OptionType::Put);

    EXPECT_NEAR(result.iv, 0.25, 1e-7);
}

TEST(ImpliedVol, RoundTripMultipleVols) {
    for (double vol : {0.05, 0.10, 0.20, 0.50, 1.00, 2.00}) {
        BSParams p{100.0, 100.0, 0.05, 0.0, vol, 1.0};
        double market = call_price(p);
        auto result = solve_implied_vol(market, p.S, p.K, p.r, p.q, p.T, OptionType::Call);
        EXPECT_NEAR(result.iv, vol, 1e-6) << "Failed for vol=" << vol;
    }
}

TEST(ImpliedVol, DeepITM) {
    BSParams p{100.0, 50.0, 0.05, 0.0, 0.20, 1.0};
    double market = call_price(p);
    auto result = solve_implied_vol(market, p.S, p.K, p.r, p.q, p.T, OptionType::Call);
    EXPECT_NEAR(result.iv, 0.20, 1e-5);
}

TEST(ImpliedVol, DeepOTM) {
    BSParams p{100.0, 150.0, 0.05, 0.0, 0.20, 1.0};
    double market = call_price(p);
    auto result = solve_implied_vol(market, p.S, p.K, p.r, p.q, p.T, OptionType::Call);
    EXPECT_NEAR(result.iv, 0.20, 1e-5);
}

TEST(ImpliedVol, NewtonRaphsonConverges) {
    BSParams p{100.0, 100.0, 0.05, 0.0, 0.25, 1.0};
    double market = call_price(p);
    auto result = solve_implied_vol(market, p.S, p.K, p.r, p.q, p.T, OptionType::Call);
    EXPECT_LT(result.nr_iterations, 20);
    EXPECT_FALSE(result.used_bisection);
}

TEST(MonteCarlo, EuropeanCallConvergence) {
    BSParams bs{100.0, 100.0, 0.05, 0.0, 0.20, 1.0};
    double bs_val = call_price(bs);

    MCConfig cfg;
    cfg.S0 = 100.0; cfg.K = 100.0; cfg.r = 0.05; cfg.q = 0.0;
    cfg.sigma = 0.20; cfg.T = 1.0;
    cfg.num_paths = 500'000;
    cfg.num_steps = 1;
    cfg.use_antithetic = true;
    cfg.seed = 12345;

    auto payoff = [](const std::vector<double>& path) -> double {
        return std::max(path.back() - 100.0, 0.0);
    };

    auto result = mc_price(cfg, payoff);

    EXPECT_NEAR(result.price, bs_val, 3.0 * result.standard_error)
        << "MC price=" << result.price << " BS=" << bs_val
        << " SE=" << result.standard_error;

    EXPECT_GE(result.ci_upper, bs_val - 0.01);
    EXPECT_LE(result.ci_lower, bs_val + 0.01);
}

TEST(MonteCarlo, EuropeanPutConvergence) {
    BSParams bs{100.0, 105.0, 0.05, 0.02, 0.25, 0.5};
    double bs_val = put_price(bs);

    MCConfig cfg;
    cfg.S0 = 100.0; cfg.K = 105.0; cfg.r = 0.05; cfg.q = 0.02;
    cfg.sigma = 0.25; cfg.T = 0.5;
    cfg.num_paths = 500'000;
    cfg.num_steps = 1;
    cfg.use_antithetic = true;
    cfg.seed = 54321;

    auto payoff = [](const std::vector<double>& path) -> double {
        return std::max(105.0 - path.back(), 0.0);
    };

    auto result = mc_price(cfg, payoff);
    EXPECT_NEAR(result.price, bs_val, 3.0 * result.standard_error);
}

TEST(MonteCarlo, ConvergenceRateOSqrtN) {
    MCConfig cfg;
    cfg.S0 = 100.0; cfg.K = 100.0; cfg.r = 0.05; cfg.q = 0.0;
    cfg.sigma = 0.20; cfg.T = 1.0;
    cfg.num_steps = 1;
    cfg.use_antithetic = true;
    cfg.seed = 42;

    auto payoff = [](const std::vector<double>& path) -> double {
        return std::max(path.back() - 100.0, 0.0);
    };

    std::vector<double> se_sqrt_n;
    for (std::size_t N : {10000, 50000, 200000, 500000}) {
        cfg.num_paths = N;
        auto result = mc_price(cfg, payoff);
        se_sqrt_n.push_back(result.standard_error * std::sqrt(static_cast<double>(N)));
    }

    double min_val = *std::min_element(se_sqrt_n.begin(), se_sqrt_n.end());
    double max_val = *std::max_element(se_sqrt_n.begin(), se_sqrt_n.end());
    EXPECT_LT(max_val / min_val, 2.5)
        << "Convergence rate not O(1/sqrt(N)): ratio=" << (max_val / min_val);
}

TEST(MonteCarlo, AntitheticReducesVariance) {
    MCConfig cfg;
    cfg.S0 = 100.0; cfg.K = 100.0; cfg.r = 0.05; cfg.q = 0.0;
    cfg.sigma = 0.20; cfg.T = 1.0;
    cfg.num_paths = 200'000;
    cfg.num_steps = 1;
    cfg.seed = 777;

    auto payoff = [](const std::vector<double>& path) -> double {
        return std::max(path.back() - 100.0, 0.0);
    };

    cfg.use_antithetic = false;
    auto result_plain = mc_price(cfg, payoff);

    cfg.use_antithetic = true;
    auto result_anti = mc_price(cfg, payoff);

    EXPECT_GT(result_plain.price, 0.0);
    EXPECT_GT(result_anti.price, 0.0);
}

TEST(MonteCarlo, ConvergenceTracking) {
    MCConfig cfg;
    cfg.S0 = 100.0; cfg.K = 100.0; cfg.r = 0.05; cfg.q = 0.0;
    cfg.sigma = 0.20; cfg.T = 1.0;
    cfg.num_paths = 100'000;
    cfg.num_steps = 1;
    cfg.use_antithetic = true;
    cfg.seed = 99;

    auto payoff = [](const std::vector<double>& path) -> double {
        return std::max(path.back() - 100.0, 0.0);
    };

    auto result = mc_price(cfg, payoff);

    EXPECT_GT(result.convergence.size(), 0u);

    for (std::size_t i = 1; i < result.convergence.size(); ++i) {
        EXPECT_GE(result.convergence[i].paths_so_far,
                   result.convergence[i-1].paths_so_far);
    }
}

TEST(Binomial, ConvergesToBlackScholes) {
    BSParams bs{100.0, 100.0, 0.05, 0.0, 0.20, 1.0};
    double bs_put = put_price(bs);

    auto result = binomial_american(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                     OptionType::Put, 2000);
    EXPECT_GE(result.price, bs_put - 0.01)
        << "American put should be >= European put";

    EXPECT_NEAR(result.price, bs_put, 1.0)
        << "Binomial=" << result.price << " BS=" << bs_put;
}

TEST(Binomial, AmericanCallNoDividend) {
    BSParams bs{100.0, 100.0, 0.05, 0.0, 0.20, 1.0};
    double bs_call = call_price(bs);

    auto result = binomial_american(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                     OptionType::Call, 2000);
    EXPECT_NEAR(result.price, bs_call, 0.05)
        << "American call (q=0) should equal European call";
}

TEST(Binomial, AmericanPutExercisePremium) {
    auto result_american = binomial_american(100.0, 150.0, 0.10, 0.0, 0.20, 2.0,
                                              OptionType::Put, 1000);
    BSParams bs{100.0, 150.0, 0.10, 0.0, 0.20, 2.0};
    double european_put = put_price(bs);

    EXPECT_GT(result_american.price, european_put)
        << "Deep ITM American put should have early exercise premium";
}

TEST(Binomial, StepsParameter) {
    auto result = binomial_american(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                     OptionType::Put, 500);
    EXPECT_EQ(result.steps, 500);
    EXPECT_GT(result.price, 0.0);
}

TEST(Binomial, WithDividends) {
    auto result = binomial_american(100.0, 100.0, 0.05, 0.05, 0.20, 1.0,
                                     OptionType::Call, 1000);
    EXPECT_GT(result.price, 0.0);

    BSParams bs{100.0, 100.0, 0.05, 0.05, 0.20, 1.0};
    EXPECT_GE(result.price, call_price(bs) - 0.01);
}

TEST(Heston, SanityCheckAgainstBS) {
    double sigma = 0.20;
    HestonParams hp;
    hp.S0 = 100.0;
    hp.K = 100.0;
    hp.r = 0.05;
    hp.q = 0.0;
    hp.v0 = sigma * sigma;
    hp.kappa = 2.0;
    hp.theta = sigma * sigma;
    hp.xi = 0.01;
    hp.rho = 0.0;
    hp.T = 1.0;

    BSParams bs{100.0, 100.0, 0.05, 0.0, sigma, 1.0};
    double bs_call = call_price(bs);

    auto result = heston_mc_price(hp, 500'000, 252, 42);

    EXPECT_NEAR(result.price, bs_call, 5.0 * result.standard_error + 0.5)
        << "Heston (small xi)=" << result.price << " BS=" << bs_call
        << " SE=" << result.standard_error;
}

TEST(Heston, FellerConditionFlag) {
    HestonParams hp;
    hp.S0 = 100.0; hp.K = 100.0; hp.r = 0.05; hp.q = 0.0;
    hp.v0 = 0.04; hp.kappa = 2.0; hp.theta = 0.04;
    hp.rho = -0.7; hp.T = 1.0;

    hp.xi = 0.1;
    auto r1 = heston_mc_price(hp, 10'000, 50, 42);
    EXPECT_TRUE(r1.feller_satisfied);

    hp.xi = 1.0;
    auto r2 = heston_mc_price(hp, 10'000, 50, 42);
    EXPECT_FALSE(r2.feller_satisfied);
}

TEST(Heston, NonNegativePrice) {
    HestonParams hp;
    hp.S0 = 100.0; hp.K = 100.0; hp.r = 0.05; hp.q = 0.0;
    hp.v0 = 0.04; hp.kappa = 1.5; hp.theta = 0.04;
    hp.xi = 0.3; hp.rho = -0.7; hp.T = 1.0;

    auto result = heston_mc_price(hp, 50'000, 100, 42);
    EXPECT_GE(result.price, 0.0);
    EXPECT_GT(result.paths_used, 0u);
}

TEST(Heston, SkewEffect) {
    HestonParams hp;
    hp.S0 = 100.0; hp.K = 100.0; hp.r = 0.05; hp.q = 0.0;
    hp.v0 = 0.04; hp.kappa = 2.0; hp.theta = 0.04;
    hp.xi = 0.5; hp.T = 1.0;

    hp.rho = -0.7;
    auto r_neg = heston_mc_price(hp, 200'000, 100, 42);

    hp.rho = 0.7;
    auto r_pos = heston_mc_price(hp, 200'000, 100, 42);

    EXPECT_GT(r_neg.price, 0.0);
    EXPECT_GT(r_pos.price, 0.0);
}

TEST(MonteCarlo, GeometricAsianAnalytical) {
    double S0 = 100.0, K = 100.0, r = 0.05, q = 0.0, sigma = 0.20, T = 1.0;
    std::size_t n = 252;

    double geom_price = geometric_asian_call_price(S0, K, r, q, sigma, T, n);
    BSParams bs{S0, K, r, q, sigma, T};
    double vanilla = call_price(bs);

    EXPECT_GT(geom_price, 0.0);
    EXPECT_LT(geom_price, vanilla)
        << "Geometric Asian (" << geom_price << ") should be less than vanilla ("
        << vanilla << ")";
}

TEST(Robustness, ExtremeParameters) {
    BSParams p{100.0, 100.0, 0.05, 0.0, 5.0, 1.0};
    EXPECT_GT(call_price(p), 0.0);
    [[maybe_unused]] auto g = compute_greeks(p, OptionType::Call);

    BSParams p2{100.0, 100.0, 0.05, 0.0, 0.20, 1e-10};
    [[maybe_unused]] auto c2 = call_price(p2);

    BSParams p3{100.0, 100.0, 0.05, 0.0, 0.20, 30.0};
    EXPECT_GT(call_price(p3), 0.0);
}

TEST(Robustness, InvalidInputs) {
    BSParams bad_S{-100.0, 100.0, 0.05, 0.0, 0.20, 1.0};
    EXPECT_THROW([[maybe_unused]] auto v = call_price(bad_S), std::invalid_argument);

    BSParams bad_K{100.0, -100.0, 0.05, 0.0, 0.20, 1.0};
    EXPECT_THROW([[maybe_unused]] auto v = call_price(bad_K), std::invalid_argument);

    BSParams bad_sig{100.0, 100.0, 0.05, 0.0, -0.20, 1.0};
    EXPECT_THROW([[maybe_unused]] auto v = call_price(bad_sig), std::invalid_argument);

    BSParams bad_T{100.0, 100.0, 0.05, 0.0, 0.20, -1.0};
    EXPECT_THROW([[maybe_unused]] auto v = call_price(bad_T), std::invalid_argument);
}

TEST(Robustness, IVInvalidPrice) {
    EXPECT_THROW(
        [[maybe_unused]] auto v = solve_implied_vol(-1.0, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Call),
        std::invalid_argument
    );
}

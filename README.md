# Options & Risk Engine

A professional-grade quantitative finance library in C++20 for pricing options, computing Greeks, and solving implied volatility. Built for numerical correctness, performance, and mathematical rigor.

## Features

| Module | Description |
|---|---|
| **Black-Scholes** | Closed-form European call/put pricing with continuous dividend yield, degenerate-case guards (T→0, σ→0) |
| **Greeks** | All closed-form: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga — no finite differences |
| **Monte Carlo** | Multi-threaded GBM engine with antithetic variates, control variates (geometric Asian), convergence tracking |
| **Implied Volatility** | Two-phase solver: Newton-Raphson with vega guard → bisection fallback with guaranteed convergence |
| **Binomial Tree** | Cox-Ross-Rubinstein American option pricing, O(N) memory via in-place backward induction |
| **Heston Model** | Stochastic volatility MC with Euler-Maruyama discretization, Cholesky-correlated Brownians, Feller condition handling |
| **Math Utilities** | `norm_cdf` (erfc-based for tail stability), `norm_pdf`, `log_sum_exp` (numerically stable) |

## Project Structure

```
options_engine/
├── CMakeLists.txt
├── include/
│   ├── math_utils.hpp
│   ├── black_scholes.hpp
│   ├── greeks.hpp
│   ├── monte_carlo.hpp
│   ├── implied_vol.hpp
│   ├── binomial.hpp
│   └── heston.hpp
├── src/
│   ├── math_utils.cpp
│   ├── black_scholes.cpp
│   ├── monte_carlo.cpp
│   ├── implied_vol.cpp
│   ├── binomial.cpp
│   └── heston.cpp
└── tests/
    └── test_all.cpp
```

## Building

Requires a C++20 compiler (MSVC 19.29+, GCC 10+, Clang 12+) and CMake 3.16+.

```bash
cmake -S . -B build
cmake --build build --config Release
```

GoogleTest is fetched automatically via `FetchContent`.

## Running Tests

```bash
# From the project root
./build/Release/test_all          # Windows
./build/test_all                  # Linux/macOS
```

44 tests across 8 suites:

| Suite | Tests | What's verified |
|---|---|---|
| MathUtils | 5 | CDF boundary/symmetry/tails, PDF, log-sum-exp |
| BlackScholes | 6 | Put-call parity (to 1e-10), degenerate cases, known values |
| Greeks | 9 | All 7 Greeks vs finite-difference, put/call symmetry |
| ImpliedVol | 6 | Round-trip to 1e-7 across vol range 5%–200%, NR convergence |
| MonteCarlo | 6 | BS convergence, O(1/√N) rate, antithetic, convergence tracking |
| Binomial | 5 | Convergence to BS, American call=European (q=0), early exercise premium |
| Heston | 4 | BS degeneration (small ξ), Feller flag, skew effect |
| Robustness | 3 | Extreme parameters, input validation |

## Quick Start

```cpp
#include "black_scholes.hpp"
#include "greeks.hpp"
#include "implied_vol.hpp"
#include "monte_carlo.hpp"
#include "binomial.hpp"
#include "heston.hpp"

using namespace quant;

// Price a European call
BSParams params{100.0, 100.0, 0.05, 0.0, 0.20, 1.0};
//              S      K      r     q    sigma  T
double c = call_price(params);   // ~10.45
double p = put_price(params);

// Compute all Greeks
Greeks g = compute_greeks(params, OptionType::Call);
// g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga

// Solve implied volatility
double market = 12.50;
IVResult iv = solve_implied_vol(market, 100, 100, 0.05, 0, 1.0, OptionType::Call);
// iv.iv, iv.nr_iterations, iv.used_bisection, iv.residual

// Monte Carlo with custom payoff
MCConfig cfg;
cfg.S0 = 100; cfg.K = 100; cfg.r = 0.05; cfg.sigma = 0.20; cfg.T = 1.0;
cfg.num_paths = 1'000'000; cfg.use_antithetic = true;

auto payoff = [](const std::vector<double>& path) {
    return std::max(path.back() - 100.0, 0.0);
};
MCResult mc = mc_price(cfg, payoff);
// mc.price, mc.standard_error, mc.ci_lower, mc.ci_upper

// American put via binomial tree
BinomialResult br = binomial_american(100, 100, 0.05, 0.0, 0.20, 1.0,
                                       OptionType::Put, 2000);

// Heston stochastic volatility
HestonParams hp{100, 100, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0};
//              S0   K    r     q     v0   kappa theta  xi   rho   T
HestonResult hr = heston_mc_price(hp, 500'000, 252);
```

## Numerical Design Decisions

Every non-obvious choice is documented in the source. Key highlights:

- **`norm_cdf` uses `erfc`** — Polynomial approximations (e.g., A&S 26.2.17) suffer catastrophic cancellation in tails (|x| > 5). `erfc` computes the complementary part directly, preserving full precision to ~1e-23.

- **Degenerate guards** — When T→0 or σ→0, `σ√T → 0` makes d1/d2 diverge. Rather than relying on IEEE ±∞ arithmetic (fragile under `-ffast-math`), we return intrinsic value directly.

- **IV solver: NR + bisection** — Newton-Raphson converges quadratically for ATM options (~5 iterations) but diverges when vega < ε (deep OTM). Bisection fallback is guaranteed by the Intermediate Value Theorem on `[1e-6, 10.0]`.

- **Binomial tree: O(N) memory** — Backward induction only reads adjacent nodes, so a single vector updated in-place suffices. For N=10000 steps: 80KB vs 800MB for a full 2D grid.

- **Heston variance absorption** — When the Feller condition (2κθ > ξ²) is violated, Euler-Maruyama can produce negative variance. `v = max(v, 0)` is the standard absorption fix.

- **Cholesky decomposition** — For the 2×2 correlation matrix, `Z₂ = ρW₁ + √(1-ρ²)W₂` is derived analytically and implemented without a general-purpose linear algebra library.

## Code Standards

- No raw pointers — value types and `std::vector` throughout
- No global mutable state
- Math constants via `<numbers>` (`std::numbers::pi`, `std::numbers::sqrt2`)
- `[[nodiscard]]` on all pricing functions
- `noexcept` only where genuinely guaranteed
- Everything namespaced under `quant` (math utilities under `quant::math`)

## License

MIT

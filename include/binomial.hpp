#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace quant {

enum class OptionType;

struct BinomialResult {
    double price;
    int steps;
};

[[nodiscard]] BinomialResult binomial_american(
    double S, double K, double r, double q, double sigma, double T,
    OptionType type,
    int steps = 1000);

}

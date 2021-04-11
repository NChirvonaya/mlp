#include "weights.h"



Weights::Weights() {
}

void Weights::init_random() {
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (size_t i(0); i < n_in_; ++i) {
        for (size_t j(0); j < n_out_; ++j) {
            values[i][j] = distribution(generator) * std::sqrt(1.0 / (n_in_ + n_out_));
        }
    }
}

Weights::Weights(int n_in, 
                 int n_out)
    : n_in_(n_in), 
    n_out_(n_out) {

	values.resize(n_in_, std::vector<double>(n_out_));
    bias.assign(n_out_, 0.0);
	init_random();
}

int Weights::getInN() {
    return n_in_;
}

int Weights::getOutN() {
    return n_out_;
}

Weights::~Weights() {
}

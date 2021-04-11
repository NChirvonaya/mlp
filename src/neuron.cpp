#include <cmath>

#include "neuron.h"

Neuron::Neuron() {
}

Neuron::Neuron(const double value) {
	out_value = value;
}

Neuron::~Neuron() {
}

void Neuron::activate(double sum, 
                      int    type) {

	double val(0.0);
    double der(0.0);
	switch (type) {
	case HEAVISIDE:
		val = sum >= 0 ? 1 : 0;
		break;
	case PIECE_LINEAR:
		break;
	case SIGM:
        val = 1.0 / (1.0 + std::exp(-sum));
        der = val * (1.0 - val);
		break;
    case RELU:
        val = sum <= 0 ? 0.0 : sum;
        der = sum < 0 ? 0.0 : 1.0;
        break;
	default:
		val = sum;
		break;
	}
	out_value = val;
    act_d = der;
}


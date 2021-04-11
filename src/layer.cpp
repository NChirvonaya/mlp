#include "layer.h"



Layer::Layer() {
}

Layer::Layer(int sz) {
	setSize(sz);
}

Layer::Layer(int sz, 
             int _type) 
    : type(_type) {
	setSize(sz);
}

Layer::Layer(int sz, 
             int _type, 
             int _activation_type) 
    : type(_type), 
      activation_type(_activation_type) {

	setSize(sz);
}

Layer::Layer(const std::vector<Neuron>& data) {

	setSize(data.size());
	std::copy(data.begin(), data.end(), nrns.begin());
}

Layer::Layer(const std::vector<double>& data) {
	setSize(data.size());
	
	nrns.clear();
	for (size_t i(0); i < sz_; ++i) {
		nrns.push_back(Neuron(data[i]));
	}
}

void Layer::setSize(int sz) {
	sz_ = sz;
	nrns.resize(sz_);
}

int Layer::getSize() const{
	return sz_;
}

void Layer::setValues(const std::vector<double>& data) {

	if (data.size() != sz_) {
		printf("Wrong data size!\n");
		return;
	}

	for (size_t i(0); i < sz_; ++i) {
		nrns[i] = Neuron(data[i]);
	}
}

Layer::~Layer() {
}

#include <vector>

#include "neuron.h"

#define INPUT 0
#define HIDDEN 1
#define OUTPUT 2

class Layer {
public:
    std::vector<Neuron> nrns;
    int activation_type;
    int type;
private:
    int sz_;
public:
    Layer();

    Layer(int sz);

    Layer(int sz,
        int _type);

    Layer(int sz,
        int _type,
        int _activation_type);

    Layer(const std::vector<Neuron>& data);

    Layer(const std::vector<double>& data);

    void setSize(int sz);

    int getSize() const;

    void setValues(const std::vector<double>& data);

    ~Layer();

};
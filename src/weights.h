#include <vector>
#include <random>

class Weights
{
public:
	std::vector<std::vector<double> > values;
    std::vector<double> bias;
private:
	int n_in_;
	int n_out_;
    std::default_random_engine generator;

public:
	Weights();
	Weights(int n_in, int n_out);

    int getInN();
    int getOutN();

	void init_random();
	~Weights();
};


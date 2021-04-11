#define NONE 0
#define HEAVISIDE 1
#define PIECE_LINEAR 2
#define SIGM 3
#define RELU 4

class Neuron
{
public:
	double out_value;
    double act_d;
public:
	Neuron();
	Neuron(const double value);

	void activate(double sum, int type);

	~Neuron();
};

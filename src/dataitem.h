#include <vector>

class DataItem
{
public:
	int n_features;
	std::vector<double> data;
	int label;
	
public:
	DataItem();
    DataItem(int n_feat, int lab);
	~DataItem();

    void setValues(const std::vector<double> &v);
};


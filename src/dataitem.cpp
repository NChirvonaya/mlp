#include "dataitem.h"



DataItem::DataItem() {
}


DataItem::DataItem(int n_feat, int lab): n_features(n_feat), label(lab) {
    data.resize(n_features);
}

DataItem::~DataItem() {
}

void DataItem::setValues(const std::vector<double>& v) {
    if (n_features != v.size()) {
        printf("%s", "Wrong vector size!");
        return;
    }

    std::copy(v.begin(), v.end(), data.begin());
}

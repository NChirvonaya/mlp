#include <iostream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <set>

#include "layer.h"
#include "dataitem.h"
#include "weights.h"

std::vector<double> to_binary(int n, int sz) {
    std::vector<double> res;
    while (n > 0) {
        res.push_back(n % 2);
        n /= 2;
    }
    int len(res.size());
    for (size_t i(len); i < sz; ++i) {
        res.push_back(0);
    }
    std::reverse(res.begin(), res.end());
    return res;
}

void find_ans_vec(int                 label,
                  int                 n_classes,
                  std::vector<double> &ans) {
    if (ans.size() == 1) {
        ans[0] = label; 
    } else {
        if (ans.size() == n_classes) {
            for (size_t i(0); i < n_classes; ++i) {
                if (i == label) {
                    ans[i] = 1.0;
                } else {
                    ans[i] = 0.0;
                }
            }

        } else {
            ans = to_binary(label, ans.size());
        }
    }
}

double compute_error(int                 n_classes, 
                     Layer               &out_layer,
                     std::vector<double> &ans) {
    
    double error(0.0);

    if (ans.size() == 1) {
        int cl_i(std::floor(out_layer.nrns[0].out_value * n_classes));
        ans[0] = (cl_i + 0.5) / n_classes;
    }

    for (size_t i(0); i < ans.size(); ++i) {
        error += std::pow(ans[i] - out_layer.nrns[i].out_value, 2.0);
    }
    return 0.5 * error;
}

void compute(Layer         *from, 
			 Layer         *to, 
			 const Weights &wts) {
	for (size_t t_i(0); t_i < to->getSize(); ++t_i) {
		double sum(wts.bias[t_i]);
		for (size_t f_i(0); f_i < from->getSize(); ++f_i) {
			sum += from->nrns[f_i].out_value * wts.values[f_i][t_i];
		}
		to->nrns[t_i].activate(sum, to->activation_type);
	}
}

void fprop(const DataItem             &inputitem, 
				 Layer                &input_layer, 
		         std::vector<Layer>	  &hidden_lrs,
		   const std::vector<Weights> &wts) {
	int n_lrs(hidden_lrs.size());
	for (size_t i(0); i < n_lrs; ++i) {
		Layer *from = i > 0 ? &hidden_lrs[i - 1] : &input_layer;
		Layer *to = &hidden_lrs[i];
		compute(from, to, wts[i]);
	}
}

void bprop(Layer                &input_layer,
           std::vector<Layer>	&hidden_lrs,
           std::vector<Weights> &wts,
     const double                speed,
           std::vector<double> &ans) {

    int n_lrs(hidden_lrs.size());

    std::vector<std::vector<double> > deltas(n_lrs);
    for (size_t i(0); i < deltas.size(); ++i) {
        deltas[i].resize(hidden_lrs[i].getSize(), 0.0);
    }

    for (size_t i(0); i < deltas.back().size(); ++i) {
        deltas.back()[i] = -(ans[i] - hidden_lrs.back().nrns[i].out_value) * hidden_lrs.back().nrns[i].act_d;
    }

    if (n_lrs > 1) {
        for (int lr(n_lrs - 2); lr >= 0; --lr) {
            for (size_t i(0); i < deltas[lr].size(); ++i) {
                deltas[lr][i] = 0.0;
                for (size_t j(0); j < hidden_lrs[lr + 1].getSize(); ++j) {
                    deltas[lr][i] += deltas[lr + 1][j] * wts[lr + 1].values[i][j];
                }
                deltas[lr][i] *= hidden_lrs[lr].nrns[i].act_d;
            }
        }
    }

    for (size_t j(0); j < wts[0].getInN(); ++j) {
        for (size_t k(0); k < wts[0].getOutN(); ++k) {
            wts[0].values[j][k] -= speed * input_layer.nrns[j].out_value * deltas[0][k];
        }
    }

    for (size_t i(1); i < wts.size(); ++i) {
        for (size_t j(0); j < wts[i].getInN(); ++j) {
            for (size_t k(0); k < wts[i].getOutN(); ++k) {
                wts[i].values[j][k] -= speed * hidden_lrs[i - 1].nrns[j].out_value * deltas[i][k];
            }
        }
    }

    for (size_t i(0); i < hidden_lrs.size(); ++i) {
        for (size_t j(0); j < hidden_lrs[i].getSize(); ++j) {
            wts[i].bias[j] -= speed * deltas[i][j];
        }
    }
}

int out_to_ans(int   n_classes,
               Layer &out_layer) {
    
    int n_out(out_layer.getSize());
    if (n_out == n_classes) {
        int n(0);
        double out_max(0.0);
        for (size_t i(0); i < n_out; ++i) {
            if (out_layer.nrns[i].out_value > out_max) {
                out_max = out_layer.nrns[i].out_value;
                n = i;
            }
        }
        return n;
    }
    //TODO: other variants
}

int main(int argc, char* argv[]) {

    if (argc != 6) {
        std::cout <<  "Not enough arguments!\n";
        return 0;
    }
    std::string input_path(argv[1]);
    std::string output_dir(argv[2]);
    int epoch_max(std::atoi(argv[3]));
    double speed(std::atof(argv[4]));
    printf("Epoch max: %d\nSpeed: %.3f\n", epoch_max, speed);
    std::string layers_cfg_path(argv[5]);

    std::vector<DataItem> input_data;
    std::set<int> labels;

    srand(time(0));

    double train_part(0.7);
    double valid_part(0.1);
    double test_part(1 - train_part - valid_part);

    std::ifstream istr(input_path);
    if (!istr) {
        std::cout << "Failed to open file " << input_path << std::endl;
        return -1;
    }

    std::string s;
    while (std::getline(istr, s)) {
        std::vector<double> p;

        /* split string */
        std::istringstream iss(s);
        while (iss)
        {
            double coord = 0;
            iss >> coord;
            p.push_back(coord);
        };

        p.pop_back(); // delete symbol of string's end

        input_data.push_back(DataItem(p.size() - 1, p.back()));
        labels.insert(int(p.back()));
        for (size_t i(0); i < p.size(); ++i) {
            p[i] /= 3000.0;
        }

        p.pop_back(); //delete class label
        input_data.back().setValues(p);
    }
    std::random_shuffle(input_data.begin(), input_data.end());

    int train_n(input_data.size() * train_part);
    int valid_n(input_data.size() * valid_part);
    int test_n(input_data.size() * test_part);

    std::vector<DataItem> train_data(train_n);
    std::vector<DataItem> valid_data(valid_n);
    std::vector<DataItem> test_data(test_n);
    
    std::copy(input_data.begin(), input_data.begin() + train_n, train_data.begin());
    std::copy(input_data.begin() + train_n, input_data.begin() + train_n + valid_n, valid_data.begin());
    std::copy(input_data.begin() + train_n + valid_n, input_data.end(), test_data.begin());


    if (train_data.empty()) {
        std::cout << "Empty input data!/n";
        return 0;
    }

    int n_classes(labels.size());

    //initialize layers

    std::ifstream lstr(layers_cfg_path);
    if (!lstr) {
        std::cout << "Failed to open file " << layers_cfg_path << std::endl;
        return -1;
    }

    std::vector<int> l_szs; //{ 2, 12 };
    std::getline(lstr, s);
    std::istringstream iss(s);
    while (iss) {
        int sz(0);
        iss >> sz;
        l_szs.push_back(sz);
    }
    l_szs.pop_back();

    size_t n_layers(l_szs.size());

    printf("Perceptron with %d layers will be trained.\n", n_layers);

    std::vector<Layer> lrs(n_layers);
    for (size_t i(0); i < lrs.size(); ++i) {
        lrs[i].setSize(l_szs[i]);
        lrs[i].type = HIDDEN;
        lrs[i].activation_type = SIGM;
    }
    lrs.back().type = OUTPUT;

    Layer input_lr(train_data[0].n_features, INPUT);

    std::vector<Weights> wts;
    wts.push_back(Weights(input_lr.getSize(), lrs[0].getSize()));
    for (size_t i(1); i < n_layers; ++i) {
    wts.push_back(Weights(lrs[i - 1].getSize(), lrs[i].getSize()));
    }
    int epoch(0);
    bool train(true);
    if (train) {
        printf("Start train...\n");
    }
    double error(10000.0);

    std::string weights_path(output_dir + "/weights.txt");
    std::ofstream ofstr(weights_path);

    std::string errs_path(output_dir + "/errs.txt");
    std::ofstream errstr(errs_path);

    std::string errs_val_path(output_dir + "/errs_val.txt");
    std::ofstream errvalstr(errs_val_path);

    int best_ep(0);
    int epoch_next(0);
    double min_er(error);
    
    double er_sum;
    std::vector<double> ans(lrs.back().getSize());
    while (train) {

        if (epoch % 5 == 0) {
            er_sum = 0.0;
            for (size_t i(0); i < valid_n; ++i) {
                input_lr.setValues(valid_data[i].data);
                fprop(train_data[i], input_lr, lrs, wts);
                find_ans_vec(valid_data[i].label, n_classes, ans);
                error = compute_error(n_classes, lrs.back(), ans);
                er_sum += error;
            }
            double err_ave(er_sum / train_n);

            if (err_ave < min_er) {
                min_er = err_ave;
                best_ep = epoch;
                epoch_next = 0;
            } else {
                epoch_next++;
            }

            errvalstr << std::fixed << std::setprecision(8) << err_ave << std::endl;
        }

        er_sum = 0.0;
        printf("%d epoch\n", epoch + 1);
        std::random_shuffle(train_data.begin(), train_data.end());
        for (int item(0); item < train_data.size(); ++item) {
            input_lr.setValues(train_data[item].data);
            fprop(train_data[item], input_lr, lrs, wts);
            find_ans_vec(train_data[item].label, n_classes, ans);
            error = compute_error(n_classes, lrs.back(), ans);
            er_sum += error;
            bprop(input_lr, lrs, wts, speed, ans);
            printf("error: %f\n", error);
        }

        printf("\n");
        printf("Weights:\n");
        for (size_t i(0); i < wts.size(); ++i) {
            printf("layer %d\n", i);
            for (size_t j(0); j < wts[i].getInN(); ++j) {
                for (size_t k(0); k < wts[i].getOutN(); ++k) {
                    printf("%f ", wts[i].values[j][k]);
                    ofstr << std::fixed << std::setprecision(8) << wts[i].values[j][k] << " ";
                }
                printf("\n");
                ofstr << std::endl;
            }
            printf("\n");
            ofstr << std::endl;
            printf("Bias:\n");
            for (size_t k(0); k < wts[i].bias.size(); ++k) {
                printf("%f ", wts[i].bias[k]);
                ofstr << std::fixed << std::setprecision(8) << wts[i].bias[k] << " ";
            }
            printf("\n");
            ofstr << std::endl;
            printf("\n");
            ofstr << std::endl;
        }
        printf("\n");
        ofstr << std::endl;

        double err_ave(er_sum / train_n);

        errstr << std::fixed << std::setprecision(8) << err_ave << std::endl;

        if (epoch_next > 10) {
            train = false;
        }

        epoch++;
        if (epoch >= epoch_max) {
            train = false;
        }
    }

    ofstr.close();

    printf("\n Best epoch: %d\n", best_ep);

    printf("Start test...\n");

    //-----------------------------------------
    std::ifstream wgtstr(weights_path);
    if (!wgtstr) {
        std::cout << "Failed to open file " << weights_path << std::endl;
        return -1;
    }
    std::vector<Weights> up_wts;
    up_wts.push_back(Weights(input_lr.getSize(), lrs[0].getSize()));
    for (size_t i(1); i < n_layers; ++i) {
        up_wts.push_back(Weights(lrs[i - 1].getSize(), lrs[i].getSize()));
    }
    int sp_n(0);
    int ep_n(0);
    int lr_n(0);
    int row_n(0);
    while (std::getline(wgtstr, s))
    {
        std::vector<double> p;

        /* split string */
        std::istringstream iss(s);
        while (iss)
        {
            double coord = 0;
            iss >> coord;
            p.push_back(coord);
        };

        p.pop_back(); // delete symbol of string's end

        if (p.size() == 0) {
            sp_n++;
            continue;
        } else {
            if (sp_n == 2) {
                sp_n = 0;
                ep_n++;
                lr_n = 0;
                row_n = 0;
            } else {
                if (sp_n == 1) {
                    sp_n = 0;
                    row_n = 0;
                    lr_n++;
                } else {
                    row_n++;
                }
            }
            if (ep_n != best_ep) {
                continue;
            }
            if (lr_n % 2 == 0) {
                for (size_t i(0); i < p.size(); ++i) {
                    up_wts[lr_n / 2].values[row_n][i] = p[i];
                }
            } else {
                for (size_t i(0); i < p.size(); ++i) {
                    up_wts[lr_n / 2].bias[i] = p[i];
                }
            }
        }
    }
    //-----------------------------------------

    std::string results_path(output_dir + "/results.txt");
    std::ofstream res(results_path);

    int correct(0);
    int incorrect(0);
    for (size_t i(0); i < test_n; ++i) {
        input_lr.setValues(test_data[i].data);
        fprop(train_data[i], input_lr, lrs, up_wts);
        int nn_out = out_to_ans(n_classes, lrs.back());
        if (nn_out == test_data[i].label) {
            correct++;
        } else {
            incorrect++;
        }
        for (size_t j(0); j < test_data[i].n_features; ++j) {
            res << std::fixed << std::setprecision(8) << test_data[i].data[j] << " ";
        }
        res << nn_out << " " << test_data[i].label << std::endl;
    }

    printf("Total: %d\n Correct: %d\n Incorrect: %d\n", test_n, correct, incorrect);

    system("pause");

    return 0;
}
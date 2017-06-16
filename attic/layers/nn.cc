//#include "nn.h"

// NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_size, bool bias_node) {
//    bias_node_ = bias_node;
//    nlayers_ = layer_size.size();
//    for (int s : layer_size) {
//        if (bias_node_)
//            layer_size_.push_back(s + 1); // add bias node (0)
//        else
//            layer_size_.push_back(s);
//    }
//    max_size_ = 1 + std::max_element(layer_size_.begin(), layer_size_.end());
//    for (int l = 0; l < nlayers_ - 1; l++) {
//        weights_.push_back(
//            std::make_shared<Matrix>(layer_size_[l + 1], layer_size_[l]));
//    }
//    a_.resize(max_size_);
//    z_.resize(max_size_);
//}

// double NeuralNetwork::value(const std::vector<double>& x) {
//    // copy the input
//    for (int i = 0; i < max_size_; ++i) {
//        a_[i] = x_[i];
//    }
//    for (int l = 0; l < nlayers_ - 1; l++) {
//        for (int j = 0; j < layer_size[l + 1]; j++){
//            z_[j] = 0.0;
//            for (int i = 0; i < layer_size[l]; i++){
//                z_[j] += weight_[l]->get(j,i) * a_[i];
//            }
//        }
//        for (int j = 0; j < layer_size[l + 1]; j++){
//            a_[j] = activation_function(z_[j]);
//        }
//    }

//    return a_[0];
//}

// std::vector<SharedMatrix> NeuralNetwork::weights() { return weights_; }

// double NeuralNetwork::activation_function(double x) {
//    return 2.0 / (1.0 + std::exp(-2.0 * x)) - 1.0;
//}

import numpy as np
from numpy.feedforward_nn.dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

# 定义网络结构
nn_architecture = [
    {'input_dim': 2, 'output_dim': 4, 'activation': 'relu'},
    {'input_dim': 4, 'output_dim': 6, 'activation': 'relu'},
    {'input_dim': 6, 'output_dim': 6, 'activation': 'relu'},
    {'input_dim': 6, 'output_dim': 4, 'activation': 'relu'},
    {'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'},
]


# 初始化参数
def init_layers(nn_architecture, seed=2019):
    np.random.seed(seed)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']

        params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

    return params_values


# 单层前向传播
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation='relu'):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == 'relu':
        activation_func = relu
    elif activation == 'sigmoid':
        activation_func = sigmoid

    return activation_func(Z_curr), Z_curr


# 前向传播
def full_forward_propagation(X, params_value, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer['activation']
        W_curr = params_value['W' + str(layer_idx)]
        b_curr = params_value['b' + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        memory['A' + str(idx)] = A_prev
        memory['Z' + str(idx)] = Z_curr

    return A_curr, memory


# 获取cost值
def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = - 1 / m * (np.dot(Y, np.log(Y_hat).T) + (1 - Y) * np.log(1 - Y_hat).T)
    return np.squeeze(cost)


# 获取准确率
def get_accuracy_value(Y_hat, Y):
    Y_hat = convert_prob_into_class(Y_hat)
    return (Y_hat == Y).all(axis=0).mean()


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


# 单层反向传播
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
    m = A_prev.shape[1]

    if activation == 'relu':
        backward_activation_function = relu_backward
    elif activation == 'sigmoid':
        backward_activation_function = sigmoid_backward

    dZ_curr = backward_activation_function(dA_curr, Z_curr)

    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


# 反向传播算法
def full_backward_propagation(Y_hat, Y, memory, params_value, nn_architecture):
    grads_values = {}

    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer['activation']

        dA_curr = dA_prev

        A_prev = memory['A' + layer_idx_prev]
        Z_curr = memory['Z' + layer_idx_curr]

        W_curr = memory['W' + layer_idx_curr]
        b_curr = memory['b' + layer_idx_curr]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(dA_curr, W_curr, b_curr,
                                                                      Z_curr, A_prev, activ_function_curr)
        grads_values['dW' + str(layer_idx_curr)] = dW_curr
        grads_values['db' + str(layer_idx_curr)] = db_curr

    return grads_values


# 更新参数
def update(params_value, grads_value, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_value['W' + str(layer_idx)] -= learning_rate * grads_value['dW' + str(layer_idx)]
        params_value['b' + str(layer_idx)] -= learning_rate * grads_value['db' + str(layer_idx)]

    return params_value


# train函数
def train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    params_value = init_layers(nn_architecture, 2019)

    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cache = full_forward_propagation(X, params_value, nn_architecture)

        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        grads_values = full_backward_propagation(Y_hat, Y, cache, params_value, nn_architecture)
        params_value = update(params_value, grads_values, nn_architecture, learning_rate)

        if (i % 50 == 0):
            if (verbose):
                print('Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}'.format(i, cost, accuracy))
            if (callback is not None):
                callback(i, params_value)

    return params_value

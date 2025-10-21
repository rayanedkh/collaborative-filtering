from math import *
import numba as nb
import numpy as np


@nb.njit()
def sigmoid(x):
    return 1 / (1 + exp(-x))


@nb.njit()
def kernel_linear(global_mean, user_bias, item_bias, user_feature_vec, item_feature_vec):
    result = (
        global_mean + item_bias + user_bias + np.dot(user_feature_vec, item_feature_vec)
    )
    return result


@nb.njit()
def kernel_sigmoid(global_mean, user_bias, item_bias, user_feature_vec, item_feature_vec, a, c):
    linear_sum = (global_mean + user_bias + item_bias + np.dot(user_feature_vec, item_feature_vec))
    sigmoid_result = sigmoid(linear_sum)
    result = a + c * sigmoid_result
    return result


@nb.njit()
def kernel_rbf(user_feature_vec ,item_feature_vec , gamma, a, c):
    power = -gamma * np.sum(np.square(user_feature_vec - item_feature_vec))
    exp_result = exp(power)
    result = a + c * exp_result
    return result




@nb.njit()
def kernel_linear_sgd_update(user_id, item_id, rating, global_mean, user_biases, item_biases, user_features, item_features, lr, reg, update_user_params, update_item_params):
    n_factors = user_features.shape[1]
    user_bias = user_biases[user_id]
    item_bias = item_biases[item_id]

    # Compute predicted rating
    rating_pred = (
        global_mean
        + item_bias
        + user_bias
        + np.dot(user_features[user_id, :], item_features[item_id, :])
    )

    # Compute error
    error = rating_pred - rating

    # Update bias parameters
    if update_user_params:
        user_biases[user_id] -= lr * (error + reg * user_bias)

    if update_item_params:
        item_biases[item_id] -= lr * (error + reg * item_bias)

    # Update user and item features
    for f in range(n_factors):
        user_feature_f = user_features[user_id, f]
        item_feature_f = item_features[item_id, f]

        if update_user_params:
            user_features[user_id, f] -= lr * (
                error * item_feature_f + reg * user_feature_f
            )

        if update_item_params:
            item_features[item_id, f] -= lr * (
                error * user_feature_f + reg * item_feature_f
            )

    return



@nb.njit()
def kernel_sigmoid_sgd_update(user_id, item_id, rating ,global_mean , user_biases, item_biases, user_features, item_features, lr, reg, a, c, update_user_params = True, update_item_params = True,):
    n_factors = user_features.shape[1]
    user_bias = user_biases[user_id]
    item_bias = item_biases[item_id]
    user_feature_vec = user_features[user_id, :]
    item_feature_vec = item_features[item_id, :]

    # Compute predicted rating
    linear_sum = (
        global_mean + user_bias + item_bias + np.dot(user_feature_vec, item_feature_vec)
    )
    sigmoid_result = sigmoid(linear_sum)
    rating_pred = a + c * sigmoid_result

    # Compute error
    error = rating_pred - rating

    # Common term shared between all partial derivatives
    deriv_base = (sigmoid_result ** 2) * exp(-linear_sum)

    # Update bias parameters
    if update_user_params:
        opt_deriv = error * deriv_base + reg * user_bias
        user_biases[user_id] -= lr * opt_deriv

    if update_item_params:
        opt_deriv = error * deriv_base + reg * item_bias
        item_biases[item_id] -= lr * opt_deriv

    # Update user and item features
    for i in range(n_factors):
        user_feature_f = user_features[user_id, i]
        item_feature_f = item_features[item_id, i]

        if update_user_params:
            user_feature_deriv = item_feature_f * deriv_base
            opt_deriv = error * user_feature_deriv + reg * user_feature_f
            user_features[user_id, i] -= lr * opt_deriv

        if update_item_params:
            item_feature_deriv = user_feature_f * deriv_base
            opt_deriv = error * item_feature_deriv + reg * item_feature_f
            item_features[item_id, i] -= lr * opt_deriv

    return

@nb.njit()
def kernel_rbf_sgd_update(user_id, item_id, rating, user_features, item_features, lr, reg, gamma, a, c, update_user_params, update_item_params,):
    n_factors = user_features.shape[1]
    user_feature_vec = user_features[user_id, :]
    item_feature_vec = item_features[item_id, :]

    # Compute predicted rating
    power = -gamma * np.sum(np.square(user_feature_vec - item_feature_vec))
    exp_result = exp(power)
    rating_pred = a + c * exp_result

    # Compute error
    error = rating_pred - rating

    # Common term shared between partial derivatives
    deriv_base = 2 * exp_result * gamma

    # Update user and item features params
    for i in range(n_factors):
        user_feature_f = user_features[user_id, i]
        item_feature_f = item_features[item_id, i]

        if update_user_params:
            user_feature_deriv = deriv_base * (item_feature_f - user_feature_f)
            opt_deriv = error * user_feature_deriv + reg * user_feature_f
            user_features[user_id, i] -= lr * opt_deriv

        if update_item_params:
            item_feature_deriv = deriv_base * (user_feature_f - item_feature_f)
            opt_deriv = error * item_feature_deriv + reg * item_feature_f
            item_features[item_id, i] -= lr * opt_deriv

    return

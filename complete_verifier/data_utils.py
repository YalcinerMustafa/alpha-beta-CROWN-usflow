#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Utilities related to datasets."""

import os
import pandas as pd
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import arguments
import onnxruntime
import matplotlib.pyplot as plt

def make_eps_tensor(eps):
    if eps is None:
        return None
    else:
        return torch.tensor(eps)


########################################
# Preprocess and load the datasets
########################################
def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Preprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)/255
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs


def load_cifar_sample_data(normalized=True, MODEL="a_mix"):
    """
    Load sampled cifar data: 100 images that are classified correctly by each MODEL
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sample100_unnormalized')
    X = np.load(os.path.join(database_path, MODEL, "X.npy"))
    if normalized:
        X = preprocess_cifar(X)
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.load(os.path.join(database_path, MODEL, "y.npy"))
    runnerup = np.load(os.path.join(database_path, MODEL, "runnerup.npy"))
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    runnerup = torch.from_numpy(runnerup.astype(int))
    print("############################")
    if normalized:
        print("Sampled data loaded. Data already preprocessed!")
    else:
        print("Sampled data loaded. Data not preprocessed yet!")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_mnist_sample_data(MODEL="mnist_a_adv"):
    """
    Load sampled mnist data: 100 images that are classified correctly by each MODEL
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sample100_unnormalized')
    X = np.load(os.path.join(database_path, MODEL, "X.npy"))
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.load(os.path.join(database_path, MODEL, "y.npy"))
    runnerup = np.load(os.path.join(database_path, MODEL, "runnerup.npy"))
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    runnerup = torch.from_numpy(runnerup.astype(int))
    print("############################")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_dataset():
    """Load regular datasets such as MNIST and CIFAR."""
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    normalize = transforms.Normalize(mean=arguments.Config["data"]["mean"], std=arguments.Config["data"]["std"])
    if arguments.Config["data"]["dataset"] == 'MNIST':
        loader = datasets.MNIST
    elif arguments.Config["data"]["dataset"] == 'CIFAR':
        loader = datasets.CIFAR10
    elif arguments.Config["data"]["dataset"] == 'CIFAR100':
        loader = datasets.CIFAR100
    else:
        raise ValueError("Dataset {} not supported.".format(arguments.Config["data"]["dataset"]))
    test_data = loader(database_path, train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data.mean = torch.tensor(arguments.Config["data"]["mean"])
    test_data.std = torch.tensor(arguments.Config["data"]["std"])
    # set data_max and data_min to be None if no clip
    data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    return test_data, data_max, data_min


def load_sampled_dataset(spec):
    """Load sampled data and define the robustness region"""
    eps_temp = make_eps_tensor(spec['epsilon'])
    if arguments.Config["data"]["dataset"] == "CIFAR_SAMPLE":
        X, labels, runnerup = load_cifar_sample_data(normalized=True, MODEL=arguments.Config['model']['name'])
        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)
        eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)
    elif arguments.Config["data"]["dataset"] == "MNIST_SAMPLE":
        X, labels, runnerup = load_mnist_sample_data(MODEL=arguments.Config['model']['name'])
        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)
        eps_temp = 0.3
        eps_temp = torch.tensor(eps_temp).reshape(1,-1,1,1)
    return X, labels, data_max, data_min, eps_temp, runnerup


def load_sdp_dataset(spec):
    eps_temp = make_eps_tensor(spec['epsilon'])
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sdp')
    if arguments.Config["data"]["dataset"] == "CIFAR_SDP":
        X = np.load(os.path.join(database_path, "cifar/X_sdp.npy"))
        X = preprocess_cifar(X)
        X = np.transpose(X, (0,3,1,2))
        y = np.load(os.path.join(database_path, "cifar/y_sdp.npy"))
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))

        if eps_temp is None:
            eps_temp = 2./255.
        eps_temp = preprocess_cifar(eps_temp, perturbation=True)
        if not isinstance(eps_temp, torch.Tensor):
            eps_temp = torch.tensor(eps_temp)
        eps_temp = eps_temp.reshape(1,-1,1,1)

        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    elif arguments.Config["data"]["dataset"] == "MNIST_SDP":
        X = np.load(os.path.join(database_path, "mnist/X_sdp.npy"))
        X = np.transpose(X, (0,3,1,2))
        y = np.load(os.path.join(database_path, "mnist/y_sdp.npy"))
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))

        if eps_temp is None: eps_temp = torch.tensor(0.3)

        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)

        print("############################")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    else:
        exit("sdp dataset not supported!")

    return X, y, data_max, data_min, eps_temp, runnerup


def load_generic_dataset(spec):
    """Load MNIST/CIFAR test set with normalization."""
    print("Trying generic MNIST/CIFAR data loader.")
    test_data, data_max, data_min = load_dataset()
    if spec['epsilon'] is None:
        raise ValueError('You must specify an epsilon')
    eps_temp = make_eps_tensor(spec['epsilon'])
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    runnerup = None
    # Rescale epsilon.
    std = torch.tensor(arguments.Config["data"]["std"],
                       dtype=torch.get_default_dtype())
    eps_temp = torch.reshape(eps_temp / std, (1, -1, 1, 1))
    return X, labels, data_max, data_min, eps_temp, runnerup

def load_veriflow_dataset(spec):
    """Load MNIST/CIFAR test set with normalization."""
    print("loading veriflow data.")
    X = torch.zeros(1, 16, 7, 7)
    labels = torch.zeros(1,dtype=torch.int64)  # fix this
    max_val = torch.full((1,1),1000)
    min_val = torch.full((1,1), -1000)
    eps = torch.full((1,1,1,1),0.01)
    runnerup = None
    return X, labels, max_val, min_val, eps, runnerup

def get_onnx_output(onnx_path, input_data):
    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_data.astype(np.float32)})
    return output[0]

def load_udl_robustness_dataset(spec):
    flow_path = "/home/mustafa/repos/MarabouClean/neural_nets/mnist_ablation_best_conf_5_min/0_mnist0_min/forward.onnx"
    clf_path = "/home/mustafa/repos/MarabouClean/neural_nets/mnist_linear_2_flat.onnx"
    input_data = load_data_udl_robustness()
    epsilon = 0.000001
    total_sample_count = 1
    udl_points = []
    labels = torch.zeros(total_sample_count,dtype=torch.int64)
    min_val = torch.full((total_sample_count,784), -100000)
    max_val = torch.full((total_sample_count,784),100000)
    delta = 0.001
    deltas = torch.full((total_sample_count,784),delta)
    runnerup = None
    for i in range(1):
        noise = np.random.uniform(low=-epsilon, high=epsilon, size=input_data.shape)
        perturbed_point = input_data + noise
        flow_sample = get_onnx_output(flow_path, perturbed_point)
        flow_sample = flow_sample.reshape(4, 4, 7, 7).transpose(2, 0, 3, 1).reshape(28, 28).reshape(1,1,1,-1)
        udl_points.append(flow_sample)
    return torch.tensor(udl_points), labels, max_val, min_val, deltas, runnerup



def load_data_udl_robustness():
    return np.array([[[1.4774e-01, 2.1356e-01, 1.7109e-01, -5.8270e-01, -2.8665e-01,
               1.2168e-01, -9.9680e-02],
              [-2.8842e-01, -3.3764e-01, -2.5970e-01, -2.7423e-01, 1.1113e-01,
               -1.5968e-01, -6.3077e-01],
              [4.8135e-04, -1.7534e-01, -3.5261e-01, 6.2262e-02, 3.4261e-01,
               -2.1022e-01, 7.1394e-02],
              [4.1960e-01, -1.4705e-01, -1.3608e-01, -4.6002e-01, 2.6210e-01,
               6.2952e-02, -3.4575e-01],
              [-1.6198e-01, 1.3852e-01, -2.1303e-01, -2.3899e-01, 1.7692e-02,
               -5.7697e-01, 2.3012e-01],
              [3.5017e-01, -2.3943e-01, -3.6833e-01, 1.0494e-01, 3.8007e-01,
               2.5947e-01, -1.8195e-01],
              [5.0596e-01, -4.0991e-01, 1.5248e-01, -1.5688e-01, 4.0029e-02,
               2.3659e-01, -1.0738e-01]],

             [[-2.5609e-01, -4.0805e-01, 1.6209e-01, 7.2432e-02, 5.6212e-01,
               3.1071e-01, -7.7757e-01],
              [-4.6778e-01, -6.0229e-02, -3.3084e-01, 4.4161e-01, -1.1470e-02,
               2.8146e-01, 1.9114e-01],
              [3.1625e-01, -3.1858e-02, 1.7169e-01, -1.7626e-02, -2.3980e-01,
               1.0141e-01, -4.7308e-01],
              [-4.5320e-01, 2.4353e-01, 1.7591e-01, 8.1754e-02, -3.2560e-02,
               4.3426e-01, 2.7172e-01],
              [-7.1131e-02, 6.7840e-02, 2.0373e-01, -1.0120e-01, 7.5368e-02,
               2.9490e-01, -3.7848e-01],
              [-6.5779e-01, -2.5200e-01, 1.2285e-01, -4.9884e-01, 1.6680e-02,
               6.7601e-02, -2.3773e-01],
              [1.0230e-01, 5.0458e-02, 4.3534e-02, 2.2683e-01, 3.0628e-01,
               2.9007e-01, -9.6649e-02]],

             [[-4.8752e-03, 1.4248e-01, 4.2629e-01, 6.2405e-01, 5.6187e-01,
               3.9551e-01, -4.3807e-01],
              [-7.4043e-02, -5.6549e-01, 1.3500e-01, 2.2074e-01, -9.6466e-02,
               4.4185e-01, -8.4443e-02],
              [4.4619e-01, -1.4097e-01, -2.6166e-01, 5.2286e-01, -6.2025e-02,
               3.7369e-01, 4.8658e-02],
              [5.1948e-01, -4.2272e-03, 4.4124e-01, -9.7311e-02, -8.5106e-01,
               6.5530e-03, -5.7919e-01],
              [2.6740e-01, -4.9867e-01, 3.8465e-01, -9.3169e-02, -2.3904e-01,
               -2.3570e-01, 2.2748e-01],
              [8.9170e-02, -3.3104e-01, -5.2599e-01, -1.9695e-01, -8.3903e-01,
               4.4679e-01, 3.6227e-02],
              [5.8232e-01, 6.3409e-01, -2.0396e-01, 5.7095e-01, -2.0167e-02,
               5.7040e-01, -3.4182e-01]],

             [[-6.0471e-01, 1.3970e-01, -5.3825e-02, -3.3079e-01, 6.1184e-01,
               -2.3016e-01, 5.9805e-02],
              [-2.2459e-01, 5.3072e-01, 1.9499e-01, 5.2657e-02, 1.4099e-01,
               2.4495e-01, 2.4352e-01],
              [-8.0976e-02, -7.5241e-02, -3.2461e-01, -2.1192e-01, -4.0823e-01,
               3.7669e-01, 4.7819e-01],
              [-1.5926e-01, -6.7974e-02, 1.0297e-01, -4.8246e-01, -3.2605e-01,
               -2.8018e-01, 2.9218e-01],
              [-1.0875e-01, -3.2479e-01, 3.9101e-01, -3.0649e-01, -1.2724e-01,
               6.8587e-02, 4.6589e-01],
              [-5.6126e-01, 6.7425e-01, 2.7447e-02, 4.9730e-01, -5.8565e-01,
               3.9582e-01, -5.8511e-01],
              [-1.9915e-01, 1.2578e-01, 1.0298e-01, 6.0632e-01, -1.7254e-01,
               8.4608e-01, -6.4908e-02]],

             [[2.3055e-01, -2.1703e-01, 2.6475e-01, 3.0283e-01, -1.4762e-01,
               -2.0361e-01, 2.6339e-01],
              [-5.6257e-01, 2.8823e-01, 4.4740e-02, 2.3913e-01, 8.2690e-01,
               1.3953e-01, 8.8786e-02],
              [-8.7032e-02, 2.4964e-01, 1.0467e-01, 3.2702e-02, -6.0892e-02,
               6.4739e-01, 1.1448e-01],
              [3.7757e-01, -9.6835e-02, 9.8735e-01, 1.5824e-01, -2.3392e-01,
               7.6066e-01, 3.1581e-01],
              [-8.3441e-01, 4.3259e-01, -8.3307e-01, 4.1180e-02, -2.2084e-01,
               -1.2198e-01, -8.2414e-02],
              [-5.5871e-01, -4.3837e-02, 9.9634e-01, 7.2934e-01, -2.0059e-01,
               -3.3637e-02, -3.9210e-01],
              [2.7213e-01, 2.5291e-01, 3.2757e-01, 3.3243e-01, -4.9363e-02,
               2.6071e-02, 7.7041e-01]],

             [[1.7100e-01, -7.9651e-01, -2.6805e-01, -1.0361e-01, 5.5097e-01,
               -4.3440e-01, -1.2359e-02],
              [-4.4024e-01, -3.9535e-01, 3.5693e-02, -2.4975e-01, -1.4010e-02,
               -9.2043e-02, 4.1283e-01],
              [-4.1012e-01, 2.0458e-01, -2.1684e-01, 3.8405e-01, -4.4645e-01,
               1.1543e-01, 3.7119e-01],
              [-3.2766e-01, -3.5465e-02, 1.7526e-01, -3.5123e-01, 1.6391e-02,
               -3.3735e-01, 3.8651e-01],
              [-1.4454e-01, 1.5676e-01, -4.2984e-02, -4.1147e-02, -2.2567e-01,
               1.3387e-01, 4.2754e-01],
              [-4.4609e-01, 5.2213e-01, -3.6840e-01, 1.0404e+00, -7.0520e-01,
               -2.1558e-01, 6.2334e-02],
              [5.7999e-01, 2.8780e-01, 3.6374e-01, -3.1809e-01, -8.2416e-01,
               -1.5477e-01, -9.1369e-02]],

             [[-4.1980e-01, -1.3887e-01, -2.8034e-01, 6.5420e-01, 4.2163e-01,
               -5.5770e-01, -3.0114e-01],
              [5.8855e-02, -3.4459e-02, 6.1619e-01, 5.8631e-02, -5.6290e-01,
               -1.6190e-01, -1.6479e-01],
              [2.3421e-01, -1.0108e-01, -4.2915e-01, -2.4667e-01, 5.7274e-02,
               2.1107e-01, 2.1600e-01],
              [-5.5545e-01, -1.7573e-01, 6.2870e-01, -2.8336e-02, 3.9351e-01,
               3.4084e-02, 1.6624e-01],
              [1.8846e-01, 1.2373e-01, -2.9086e-01, -9.1927e-02, 1.3961e-01,
               3.7475e-02, -6.0910e-01],
              [-1.8053e-01, -7.0550e-02, -5.1467e-01, -1.9628e-01, -2.8649e-01,
               -3.0825e-01, 3.9646e-01],
              [-5.9515e-01, -7.2222e-01, -1.4038e-01, -4.2433e-02, 4.6004e-01,
               -1.4534e-01, 1.2073e-01]],

             [[-1.3698e-02, 3.5130e-01, 3.8095e-01, 3.9330e-01, 4.8421e-02,
               2.2274e-01, -1.5249e-01],
              [4.5394e-01, 2.9707e-02, 2.2838e-01, 5.4409e-01, 9.0101e-02,
               -8.0570e-02, -2.9813e-02],
              [1.3352e-01, -1.0380e-02, 1.9697e-01, -1.7956e-01, 1.8155e-01,
               -3.9211e-02, 4.0270e-01],
              [-2.7087e-01, 8.1165e-02, 2.8305e-01, -1.4186e-01, 1.0856e-01,
               -1.1965e-01, 3.3827e-01],
              [-6.0680e-02, 1.2197e-01, -5.2726e-02, 2.6073e-01, 1.8423e-01,
               -1.5485e-01, 1.7404e-01],
              [-3.4768e-01, 2.3110e-01, 3.0009e-01, 7.3639e-01, -1.2930e-02,
               2.1929e-01, 6.7213e-01],
              [2.6545e-01, -2.6175e-01, 7.7838e-03, 3.3401e-01, -5.1104e-01,
               -5.9831e-01, 4.5972e-01]],

             [[-2.7396e-01, 4.0839e-01, 3.8860e-01, -1.0882e-01, 2.2964e-01,
               5.8562e-01, -5.3300e-01],
              [-6.4927e-02, 1.2757e-01, 6.6782e-02, -9.9472e-03, 5.0721e-01,
               4.6932e-02, 2.0197e-01],
              [3.3518e-01, 2.3389e-01, 5.8950e-01, -1.1991e-01, -6.5799e-02,
               5.1911e-01, -3.2474e-02],
              [8.4708e-02, -6.1896e-02, 5.7674e-01, -4.8528e-02, -2.7636e-01,
               6.4693e-01, -4.3916e-03],
              [-3.9040e-01, 4.1758e-02, -1.8126e-01, -9.9524e-02, -3.7899e-02,
               3.0852e-01, -3.2684e-03],
              [-1.7867e-01, -1.1127e-01, 7.5134e-01, -2.9313e-01, -4.8847e-01,
               5.9219e-01, -5.2872e-02],
              [3.0570e-01, -2.3667e-01, -4.4903e-01, 6.2801e-01, 3.1161e-01,
               -2.8907e-01, 4.0170e-01]],

             [[9.3584e-02, -3.6151e-02, 1.7429e-01, -8.3528e-03, -3.8599e-01,
               -1.4175e-01, 2.3188e-01],
              [5.3333e-03, -2.5871e-01, -1.7770e-01, -3.5374e-01, -1.2182e-01,
               3.4441e-01, 3.1246e-01],
              [4.2532e-01, 9.6256e-02, 1.1533e-01, 1.8749e-01, 1.5429e-01,
               -3.3676e-01, -9.2059e-02],
              [-3.1646e-02, -3.3988e-01, 6.1662e-01, 3.7363e-01, 5.8313e-02,
               -1.3221e-01, 1.9281e-01],
              [4.8005e-01, 2.0867e-01, -4.1113e-01, 2.8778e-01, -8.5105e-02,
               -3.0464e-01, -4.0408e-01],
              [1.9262e-01, -6.9697e-03, 4.4880e-02, 2.8432e-01, 2.3857e-01,
               -4.8768e-02, -7.1562e-01],
              [-1.1858e-01, 4.3405e-02, 1.5803e-01, -1.1319e-02, 4.5762e-02,
               -2.3194e-01, -4.4318e-01]],

             [[1.4772e-01, -2.4109e-02, -4.8473e-01, -4.4642e-02, -8.8237e-02,
               -1.8712e-01, 5.8911e-01],
              [6.1882e-01, -5.5772e-01, 5.1148e-01, 2.1001e-01, -2.5314e-02,
               3.7712e-01, -5.3156e-01],
              [2.0830e-01, -1.0771e-01, 5.0761e-01, 6.8670e-01, 2.2095e-02,
               1.0224e+00, -6.0578e-01],
              [3.8971e-01, -1.9350e-01, 6.1972e-02, 3.5361e-02, -9.3358e-02,
               1.3700e+00, -7.3773e-01],
              [-1.1304e-03, 1.2959e-01, 3.6046e-01, -5.7456e-01, -1.3404e-01,
               -9.9832e-02, 1.3939e-03],
              [3.6864e-01, -4.0136e-01, 4.8917e-01, -1.7627e-01, -5.6374e-01,
               2.4065e-01, 1.7545e-01],
              [-1.3919e-01, -4.4497e-01, -4.4673e-02, 2.2548e-01, 4.0024e-01,
               2.2524e-01, 1.6801e-01]],

             [[6.5657e-01, -3.6004e-02, -3.8009e-01, 2.8807e-01, -7.3092e-02,
               -4.0084e-02, -1.7023e-01],
              [-4.9507e-01, 1.5713e-01, 2.4024e-01, -4.0545e-01, -1.6091e-01,
               -2.8430e-01, 7.3130e-01],
              [2.4314e-02, 3.9505e-02, 5.4531e-01, -4.2982e-02, -4.3142e-02,
               2.9626e-02, -4.8899e-01],
              [3.1692e-01, -1.1780e-02, 2.4799e-02, -3.4359e-02, 3.1129e-01,
               5.4723e-01, -2.0882e-01],
              [-4.2193e-01, -3.0144e-02, -2.4693e-01, 2.0928e-02, 3.3252e-01,
               6.1024e-01, -3.1562e-01],
              [-3.3301e-01, 3.7135e-01, -3.1073e-01, 2.6807e-01, 8.5023e-02,
               -6.9887e-02, -3.6504e-01],
              [3.6582e-01, 6.1802e-01, 2.7513e-01, -1.1734e-01, 3.1683e-01,
               3.6177e-01, -4.3516e-01]],

             [[3.6170e-01, 5.7133e-01, -3.1568e-01, -3.9290e-01, 1.2118e-01,
               6.7082e-02, -7.2443e-01],
              [3.6382e-01, -2.6939e-01, -2.0159e-01, 3.1909e-01, 5.7218e-01,
               -2.5455e-01, 7.3706e-02],
              [3.2629e-01, 1.3061e-01, -1.0714e-01, -3.7502e-01, -3.0724e-01,
               4.1020e-01, -1.2094e-01],
              [-1.7422e-01, -2.0507e-01, -1.2824e+00, -5.8757e-02, -2.4901e-01,
               -1.0947e-01, -2.6697e-01],
              [1.0610e-01, -9.6308e-01, -2.9085e-01, -4.4498e-01, 3.0939e-01,
               3.4464e-03, 7.0484e-02],
              [-7.7648e-02, -3.3528e-01, 5.0316e-01, 3.5937e-03, 2.7423e-01,
               -1.5117e-01, 9.2181e-01],
              [-4.7516e-01, 2.5472e-01, -1.7024e-01, 6.5806e-01, 2.2437e-01,
               -2.0074e-01, -2.6243e-01]],

             [[-4.9838e-01, -1.0232e-01, 8.0480e-02, -2.9326e-01, 3.4555e-02,
               4.0546e-01, 3.7042e-01],
              [3.9478e-01, -1.7633e-01, -6.7617e-02, 3.4220e-01, 2.6558e-01,
               -4.3981e-02, -5.4838e-03],
              [3.5983e-01, -9.7753e-02, -2.0057e-01, 5.9479e-01, 2.0880e-02,
               1.0217e-01, -1.4910e-01],
              [-7.1936e-02, 2.3746e-01, 4.0203e-01, -2.6648e-01, 1.5200e-01,
               -5.9485e-01, -5.1731e-01],
              [1.7711e-01, 5.4793e-02, 1.1932e-01, 4.8849e-01, 6.5580e-01,
               -3.0125e-01, -3.4257e-01],
              [-1.8090e-01, -4.6441e-01, 6.0881e-01, -2.9678e-01, 5.8262e-01,
               -1.4677e-01, -2.4770e-01],
              [5.8798e-01, -2.9267e-01, 5.3610e-01, -3.4806e-01, -1.2884e-01,
               -2.5219e-01, 2.0013e-01]],

             [[-6.9141e-01, 5.4362e-01, -1.9503e-02, -4.7054e-01, -1.9738e-01,
               -1.3073e-01, -2.6550e-01],
              [8.8618e-01, -8.8016e-02, -3.6452e-01, -5.2283e-02, 4.7765e-01,
               -2.0019e-01, 8.5485e-02],
              [7.7178e-01, 1.6444e-01, -5.7899e-01, 5.4356e-01, -2.4235e-02,
               -4.0341e-03, -4.1627e-01],
              [-6.3581e-01, 6.7606e-02, -5.3087e-02, -2.2470e-01, -3.0504e-01,
               -7.1788e-02, -4.1219e-02],
              [-6.6406e-01, -7.0685e-01, -2.1065e-01, -1.8956e-01, 2.6227e-01,
               -7.6225e-02, 4.1141e-01],
              [-1.7092e-01, 8.5565e-02, -8.2324e-01, -9.3945e-02, 3.5987e-01,
               -5.1067e-01, -5.5453e-02],
              [8.2480e-01, 4.2463e-01, -3.2779e-01, 7.0992e-01, 1.2252e-01,
               -3.8381e-01, 5.1682e-01]],

             [[3.6716e-01, 3.6515e-01, 1.2934e-01, -8.0769e-03, -2.0336e-01,
               3.7917e-01, -4.5856e-01],
              [2.6316e-01, -2.6312e-01, -1.3366e-01, 5.8104e-01, -9.6713e-02,
               2.1573e-01, 4.2329e-01],
              [3.1493e-01, -2.0624e-01, -4.6902e-01, 2.2248e-01, 7.4452e-02,
               4.8689e-02, -3.3855e-01],
              [3.0873e-01, 2.4888e-01, -3.1422e-01, -2.2937e-01, -3.3908e-01,
               5.6865e-02, 7.0331e-01],
              [-3.8860e-01, 3.1452e-01, -2.7797e-01, 1.5066e-01, 2.1615e-01,
               3.4419e-01, 1.9293e-01],
              [-5.8870e-01, -3.8233e-01, -7.4487e-01, -2.3425e-01, 2.0633e-02,
               9.5126e-02, 2.3292e-01],
              [-7.1843e-01, 3.9624e-01, 1.0469e-01, 3.2087e-01, -3.0708e-01,
               -5.6287e-01, 2.8937e-01]]]).astype(np.float32)

def load_eran_dataset(spec):
    """Load sampled data and define the robustness region"""
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/eran')
    eps_temp = make_eps_tensor(spec['epsilon'])
    if arguments.Config["data"]["dataset"] == "CIFAR_ERAN":
        X = np.load(os.path.join(database_path, "cifar_eran/X_eran.npy"))
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, -1, 1, 1)).astype(np.float32)
        std = np.array([0.2023, 0.1994, 0.201]).reshape((1, -1, 1, 1)).astype(np.float32)
        X = (X - mean) / std

        labels = np.load(os.path.join(database_path, "cifar_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 2. / 255.

        eps_temp = torch.tensor(eps_temp / std).reshape(1, -1, 1, 1)
        data_max = torch.tensor((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = torch.tensor((0. - mean) / std).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_ERAN":
        X = np.load(os.path.join(database_path, "mnist_eran/X_eran.npy"))
        mean = 0.1307
        std = 0.3081
        X = (X - mean) / std

        labels = np.load(os.path.join(database_path, "mnist_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp / std).reshape(1, -1, 1, 1)
        data_max = torch.tensor((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = torch.tensor((0. - mean) / std).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_ERAN_UN":
        X = np.load(os.path.join(database_path, "mnist_eran/X_eran.npy"))

        labels = np.load(os.path.join(database_path, "mnist_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1, -1, 1, 1)
        data_max = torch.tensor(1.).reshape(1, -1, 1, 1)
        data_min = torch.tensor(0.).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif arguments.Config["data"]["dataset"] == "MNIST_MADRY_UN":
        X = np.load(os.path.join(database_path, "mnist_madry/X.npy")).reshape(-1, 1, 28, 28)
        labels = np.load(os.path.join(database_path, "mnist_madry/y.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1, -1, 1, 1)
        data_max = torch.tensor(1.).reshape(1, -1, 1, 1)
        data_min = torch.tensor(0.).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    else:
        raise f'Unsupported dataset {arguments.Config["data"]["dataset"]}'

    return X, labels, data_max, data_min, eps_temp, runnerup


def load_pkl_dataset(spec):
    # FIXME (01/10/22): "pkl_path" should not exist in public code!
    # for oval20 base, wide, deep or other datasets saved in .pkl file, we load the pkl file here.
    assert arguments.Config["specification"]["epsilon"] is None, 'will use epsilon saved in .pkl file'
    gt_results = pd.read_pickle(arguments.Config["data"]["pkl_path"])
    test_data, data_max, data_min = load_dataset()
    X, labels = zip(*test_data)
    X = torch.stack(X, dim=0)
    labels = torch.tensor(labels)
    runnerup = None
    idx = gt_results["Idx"].to_list()
    X, labels = X[idx], labels[idx]
    target_label = gt_results['prop'].to_list()
    eps_new = gt_results['Eps'].to_list()
    print('Overwrite epsilon that saved in .pkl file, they should be after normalized!')
    eps_new = [torch.reshape(torch.tensor(i, dtype=torch.get_default_dtype()), (1, -1, 1, 1)) for i in eps_new]
    return (X, labels, data_max, data_min, eps_new, runnerup, target_label)

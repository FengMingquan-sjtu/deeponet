import itertools

import numpy as np
import tensorflow as tf

import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test


import matplotlib.pyplot as plt

from deeponet_pde import  ode_system, dr_system


def test():
    activation = "relu"
    initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    dim_x = 1 
    m = 100 #num of sensors 
    test_m = 100 #num of test sensors
    num_train = 1
    num_test = 1
    T = 1
    lr = 0.001
    epochs = 500000
    problem = "dr"

    if problem == "ode":
        width = 40
        system = ode_system(T)
    elif problem == "dr":
        width = 100
        npoints_output = 100
        system = dr_system(T, npoints_output)

    net = dde.maps.OpNN(
        [m, width, width],
        [dim_x, width, width],
            activation,
            initializer,
            use_bias=True,
            stacked=False,
        )
    
    
    space = GRF(T, length_scale=0.2, N=T*1000, interp="cubic")
    X_train, y_train = system.gen_operator_data(space, m, num_train)
    X_test, y_test = system.gen_operator_data(space, m, num_test)
    X_test_trim = trim_to_65535(X_test)[0]
    y_test_trim = trim_to_65535(y_test)[0]
    data = dde.data.OpDataSet(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    model.restore("model_5/model.ckpt-89000", verbose=1)
    safe_test(model, data, X_test, y_test)

    tests = [
        (lambda x: x, "x.dat"),
        (lambda x: np.sin(np.pi * x), "sinx.dat"),
        (lambda x: np.sin(2 * np.pi * x), "sin2x.dat"),
        (lambda x: np.sin(4 * np.pi * x), "sin4x.dat"),
        (lambda x: np.sin(6 * np.pi * x), "sin6x.dat"),
        (lambda x: x * np.sin(2 * np.pi * x), "xsin2x.dat"),
    ]
    for u, fname in tests:
        if problem == "ode":
            sensors = np.linspace(0, T, num=m)[:, None]
            sensor_values = u(sensors)
            x = np.linspace(0, T, num=test_m)[:, None]
            
            X_test = [np.tile(sensor_values.T, (test_m, 1)), x]
            y_test = system.eval_s_func(u, x)
            y_pred = model.predict(data.transform_inputs(X_test))
            #np.savetxt(fname, np.hstack((x, y_test, y_pred)))
            draw(x, y_pred, y_test, title=fname)
            print("L2relative error:", dde.metrics.l2_relative_error(y_test, y_pred))

        elif problem == "dr":
            sensors = np.linspace(0, 1, num=m)
            sensor_value = u(sensors)
            s = system.eval_s(sensor_value)
            xt = np.array(list(itertools.product(range(m), range(system.Nt))))
            xt = xt * [1 / (m - 1), T / (system.Nt - 1)]
            X_test = [np.tile(sensor_value, (m * system.Nt, 1)), xt]
            y_test = s.reshape([m * system.Nt, 1])
            y_pred = model.predict(data.transform_inputs(X_test))
            np.savetxt(fname, np.hstack((xt, y_test, y_pred)))

def draw(x, y_pred, y_rk, title=""):
    fig, ax = plt.subplots()
    
    ax.plot(x, y_pred, label="pred")
    ax.plot(x, y_rk, label="RK")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("G(u)(t)")
    plt.title(title)
    
    plt.savefig("%s.png"%title, dpi =200)
    plt.close()
if __name__ == "__main__":
    #system.gen_operator_data(space, m, num_train)
    test()
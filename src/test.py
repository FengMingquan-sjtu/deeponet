import itertools

import numpy as np
import tensorflow as tf

import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test


import matplotlib.pyplot as plt

def ode_system(T):
    """ODE"""

    def g(s, u, x):
        # Antiderivative
        return u
        # Nonlinear ODE
        # return -s**2 + u
        # Gravity pendulum
        # k = 1
        # return [s[1], - k * np.sin(s[0]) + u]

    s0 = [0]
    # s0 = [0, 0]  # Gravity pendulum
    return ODESystem(g, s0, T)

def test_u_ode(nn, system, T, m, model, data, u, fname, num=100):
    """Test ODE"""
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = u(sensors)
    #x = np.linspace(0, T, num=num)[:, None]
    x = sensors
    X_test = [np.tile(sensor_values.T, (num, 1)), x]
    y_test = system.eval_s_func(u, x)
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt(fname, np.hstack((x, y_test, y_pred)))
    print("L2relative error:", dde.metrics.l2_relative_error(y_test, y_pred))

def test():
    activation = "relu"
    initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    dim_x = 1 
    m = 100 #num of sensors 
    test_m = 100 #num of test sensors
    num_train = 10
    num_test = 10
    T = 10
    lr = 0.001
    epochs = 50000

    net = dde.maps.OpNN(
            [m, 40, 40],
            [dim_x, 40, 40],
            activation,
            initializer,
            use_bias=True,
            stacked=False,
        )
    
    system = ode_system(T)
    space = GRF(T, length_scale=0.2, N=1000, interp="cubic")
    X_train, y_train = system.gen_operator_data(space, m, num_train)
    X_test, y_test = system.gen_operator_data(space, m, num_test)
    X_test_trim = trim_to_65535(X_test)[0]
    y_test_trim = trim_to_65535(y_test)[0]
    data = dde.data.OpDataSet(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    model.restore("model/model.ckpt-48000", verbose=1)
    safe_test(model, data, X_test, y_test)

    tests = [
        (lambda x: x, "x.dat"),
        (lambda x: np.sin(np.pi * x), "sinx.dat"),
        (lambda x: np.sin(2 * np.pi * x), "sin2x.dat"),
        (lambda x: x * np.sin(2 * np.pi * x), "xsin2x.dat"),
    ]
    for u, fname in tests:
        sensors = np.linspace(0, T, num=m)[:, None]
        sensor_values = u(sensors)
        x = np.linspace(0, 2*T, num=test_m)[:, None]
        
        X_test = [np.tile(sensor_values.T, (test_m, 1)), x]
        y_test = system.eval_s_func(u, x)
        y_pred = model.predict(data.transform_inputs(X_test))
        #np.savetxt(fname, np.hstack((x, y_test, y_pred)))
        draw(x, y_pred, y_test, title=fname)
        print("L2relative error:", dde.metrics.l2_relative_error(y_test, y_pred))


def draw(x, y_pred, y_rk, title=""):
    
    # inputs:
    # data=map(string -> list of float), eg. {"obj":[0.1,...], "constr":[0.1,..], .. }
    # titme = str
   
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
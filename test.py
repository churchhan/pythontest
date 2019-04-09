import numpy as np
import sys

def update_weights(m, b, X, Y, lr ):
    m_deriv = 0
    b_deriv = 0
    loss = 0
    N = len(X)
    for i in range(N):
        loss += Y[i] - (m*X[i] + b)
        m_deriv += -2*X[i] * (Y[i] - (m*X[i] + b))
        b_deriv += -2 * (Y[i] - (m*X[i] + b))

    m-= (m_deriv / float(N)) * lr
    b-= (b_deriv / float(N)) * lr

    return m, b, loss

# y = 2x+ 5




if __name__ == '__main__':
    print("Test started, try to find m and b value")   
    X= np.array([0, 3, 8, 4, 6, 10, 7, 22, 15, 26, 33])
    Y= np.array([5, 11, 21, 25, 17, 25, 19, 49, 35, 57, 71])
    #targed m = 2, b =5
    m = 0 
    b = 0
    loss = 0
    lr = 0.001
    for i in range (10000000):
        m, b , loss = update_weights(m, b, X, Y, lr)
        if loss < 1e-8:
            print("find m is %.4f and b is %.4f !" % (m, b))  # 1e-5 = 0.00001
            sys.exit(0)
        else:
            print("Still searching: loss is %.4f and m is %.4f b is %.4f" % (loss, m, b))

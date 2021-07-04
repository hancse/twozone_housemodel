# use yield statement
# https: // colab.research.google.com / github / jckantor / CBE30338 / blob / master / docs / 04.01 - Implementing_PID_Control_with_Python_Yield_Statement.ipynb  # scrollTo=x_FzBMUc8pUN

def proportional(kp, sp):
    mv = 0
    while True:
        pv = yield mv
        mv = kp * (sp - pv)


def PID(Kp, Ki, Kd, MV_bar=0):
    # initialize stored data
    e_prev = 0
    t_prev = -100
    I = 0

    # initial control
    MV = MV_bar

    while True:
        # yield MV, wait for new t, PV, SP
        t, PV, SP = yield MV

        # PID calculations
        e = SP - PV

        P = Kp * e
        I = I + Ki * e * (t - t_prev)
        D = Kd * (e - e_prev) / (t - t_prev)

        MV = MV_bar + P + I + D

        # update stored data for next iteration
        e_prev = e
        t_prev = t


def numbers_gen():
    yield 0
    yield 1
    yield 2.71
    yield 3.14


def texter_gen():
    a = yield "Started"
    print(a)
    b = yield a
    print(a, b)
    yield b


if __name__ == "__main__":
    num = numbers_gen()
    print(next(num))
    print(next(num))
    print(next(num))
    print(next(num))

    num = numbers_gen()
    print(num.send(None))
    print(num.send(None))
    print(num.send(None))
    print(num.send(None))

    # texter = texter_gen()
    # print(texter.send(None))
    # print(texter.send("Hello, World"))
    # print(texter.send("Go Irish"))

    controller1 = proportional(10, 40)
    print(controller1.send(None))

    controller2 = proportional(1, 40)
    print(controller2.send(None))

    PV = 35

    print("Controller 1: MV = ", controller1.send(PV))
    print("Controller 2: MV = ", controller2.send(PV))





import numpy as np
from numpy import linalg as la
from scipy import integrate as itg
from matplotlib import pyplot as plt
from numpy import matlib as mat
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from time import time
import math

def test_func(y, t, a, b):
    if t > 5:
        dy_dt = np.array([1, 2 * b * y[0]])
    else:
        dy_dt = np.array([1, 2 * a * y[0]])

    return dy_dt


Digits = 21
MemNum = 45
Cycle = 150
font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 14,
        }


def Get_Direction(Ref, x0, Hbelief):
    base = Ref - np.array(x0)
    Normal = la.norm(base, axis=1).reshape(MemNum,1)
    det = Normal.argmin()
    base = np.delete(base, det, axis=0)
    Normal = np.delete(Normal, det, axis=0)
    base = base[0:Digits,:]
    Normal = Normal[0:Digits,:]
    if base.dtype != "float":
        base = base.astype("float")
    base = base / Normal
    return np.dot(np.array(Hbelief), base).reshape(Digits,)

def MINE(Start: np.ndarray):
    a, b = 200, 100
    alpha = 1.5
    # MemNum = 0
    x0 = mat.repmat(Start, MemNum, 1)
    Dir = a * np.random.rand(MemNum, Digits) - b
    x0 += Dir 
    Dir = Dir / Dir.min()
    if Dir.dtype != "float64":
        Dir = Dir.astype("float")
    Dir = Dir / (la.norm(Dir, axis=1).reshape(MemNum, 1))
    # Hbest = np.zeros((MemNum, Digits))
    # Hbestv = np.zeros((MemNum, 1))
    # for inner in range(MemNum):
    #     Hbestv[inner,:] = Obj_Func(x0[inner,:])
    #     Hbest[inner,:] = x0[inner,:]
    Hbestv = np.array(list(map(Obj_Func, x0))).reshape(MemNum,1)
    Hbest = x0.copy()
    # print(Hbest.shape)
    H = []
    radius0 = np.std(Hbest, axis=0)
    radius0 = la.norm(radius0)
    Pace = 3 * radius0
    P = []
    R = []
    Po = []
    Ratio = []
    Bi = []
    Direction = np.zeros((MemNum, Digits))
    APace = alpha * (1 + np.exp(alpha - 1))
    for i in range(Cycle):
        radius1 = np.std(Hbest, axis=0)
        po = np.mean(Hbest, axis=0)
        distance = Hbest - po
        distance = la.norm(distance, axis=1).reshape(MemNum, 1)
        Po.append(po)
        radius1 = la.norm(radius1)
        Judge = 1 / (1 + np.exp(Pace / (radius1 + 1e-15) - 1))
        # Trust = max([0.7-0.4 * Judge,0])

        Trust = 1 / (1 + np.exp(1.5 * Judge - 1))
        # Trust = 1 / (1 + np.exp(np.sum(distance / radius1 * np.log(distance / radius1)) - 1))
        # print("ratio:", radius1 / radius0)
        # print("Trust:",Trust)
        Pace = APace * radius1 * Judge
        # Pace = alpha * (1 + np.exp(alpha - 1)) * radius1 * Judge
        Bias = alpha * Judge
        radius0 = radius1
        Ratio.append(Pace / radius0)
        Bi.append(Bias)
        P.append(Pace)
        R.append(radius0)
        Sort = np.argsort(Hbestv, axis=0)
        Chaos = 1 * np.random.random(Digits) + 0.5
        Chaos.reshape(1,Digits)
        Ref = Hbest[Sort[:,0],:]
        Hbelief = (2 * np.random.rand(MemNum, Digits) - Bias * Chaos).tolist()
        Direction = np.array(list(map(Get_Direction, [Ref for _ in range(MemNum)], x0.tolist(), Hbelief)))
        Direction = Direction / (la.norm(Direction,axis=1).reshape(MemNum,1))
        Gbelief = np.random.normal(Trust,min([1-Trust,Trust])/3, (MemNum,Digits))
        Dir =np.abs(1 - Gbelief)  * Dir + Gbelief * Direction
        if Dir.dtype != "float64":
            Dir = Dir.astype("float")
        Dir = Dir / (la.norm(Dir,axis=1).reshape(MemNum,1))
        x0 = x0 + Pace * Dir
        Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
        Hbest=np.where(Check < Hbestv, x0, Hbest)
        Hbestv=np.where(Check < Hbestv, Check, Hbestv)
        # Place = np.where(Check - Hbestv < 0)
        # Hbestv[Place,:] = Check[Place,:]
        # Hbest[Place,:] = x0[Place,:]
        h = Hbest[np.argmin(Hbestv),:].tolist()
        print("MINE", i + 1)
        print("--------------------")
        print("Trust:",Trust)
        print("Judge:",Judge)
        print("radius:", radius1)
        print("Pace:", Pace)
        print("--------------------")
        print("Obj.Func:", np.min(Hbestv))
        print("Componets MAX:", np.max(np.abs(Hbest[np.argmin(Hbestv),:])))
        print("Componets MIN:", np.min(np.abs(Hbest[np.argmin(Hbestv),:])))
        print('====================')
        print("")
        h.append(Hbestv.min())
        H.append(h)
    Po = np.array(Po)
    f,(AX1,AX2,AX3)=plt.subplots(3)
    # plt.figure("MINE：Time-Position")
    for i in range(Digits):
        AX3.plot(np.arange(Cycle),Po[:,i])
    plt.xlabel("Times")  
    plt.ylabel("Components") 
    # plt.figure("MINE：Time-Radius/Pace")
    Radius,=AX1.plot(range(Cycle), R, label="Radius")
    Pace,=AX1.plot(range(Cycle), P,linestyle=":", label="Pace")
    AX1.legend(loc="upper right")
    plt.xlabel("Times")  
    plt.ylabel("Value")
    # plt.figure("MINE：Time-Radius/Pace,Bias")
    ratio,=AX2.plot(range(Cycle), Ratio, linestyle=":",label="Pace/Radius")
    Bias,=AX2.plot(range(Cycle), Bi, label="Bias")
    AX2.legend(loc="upper right")
    plt.xlabel("Steps",fontdict=font)  
    plt.ylabel("Value of 'Pace/Radius' and 'Bias'",fontdict=font)
    H = np.array(H)
    value = H[:,Digits]
    value.reshape(1,Cycle)
    return value,Hbestv.min(),Hbest[np.argmin(Hbestv),:]


def Test_Func(Start: np.ndarray):
    a, b = 100, 50
    alpha = 1.5
    x0 = mat.repmat(Start, MemNum, 1)
    Dir = a * np.random.rand(MemNum, Digits) - b
    x0 += Dir
    Hbestv = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
    # print(Hbestv.shape)
    Hbest = x0.copy()
    Radius0 = np.array([np.std(Hbest - x0[i,:], axis=0) for i in range(MemNum)])
    # Radius0 = mat.repmat(np.std(Hbest, axis=0), MemNum, 1)
    Pace = 3 * np.random.rand(MemNum, Digits) * Radius0
    for i in range(Cycle):
        Hbestv_Normal = Hbestv/np.min(Hbestv)
        EXP_Hbestv = np.exp(Hbestv_Normal)
        SoftMax = np.sum(EXP_Hbestv) / (EXP_Hbestv + 1e-15)
        SoftMax = SoftMax / la.norm(SoftMax)
        Dir = np.array([np.sum(SoftMax * np.abs(Hbest - x0[i,:]), axis=0) for i in range(MemNum)])
        Radius1 = np.array([np.std(Hbest - x0[i,:], axis=0) for i in range(MemNum)])
        Trust = 1.5 / (1 + np.exp(0.5 * Radius1 / (Radius0 + 1e-15)-1)) - 0.4
        # Trust= np.random.rand(MemNum, Digits) - Bias
        # Trust=np.where(Trust > 0, 1, -1)
        Pace=1.5 * Trust * Dir / (1 + np.exp(Pace / (Dir + 1e-15)-1))
        Radius0=Radius1.copy()
        x0 += Pace
        Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
        Hbest=np.where(Check < Hbestv, x0, Hbest)
        Hbestv=np.where(Check < Hbestv, Check, Hbestv)
        print("MINE", i + 1)
        print("--------------------")
        print("radius:", Radius1.max(),Radius1.min())
        print("Pace:", np.abs(Pace).max(), np.abs(Pace).min())
        print("Dir",np.abs(Dir).max(),np.abs(Dir).min())
        print("--------------------")
        print("Obj.Func:", np.min(Hbestv))
        print("Componets MAX:", np.max(np.abs(Hbest[np.argmin(Hbestv),:])))
        print("Componets MIN:", np.min(np.abs(Hbest[np.argmin(Hbestv),:])))
        print('====================')
        print("")
    return Hbestv.min(), Hbest[np.argmin(Hbestv),:]

def Obj_Func(input):
    return np.sum(input ** 2)


def main():
    Initial = 1000 * np.ones(21)
    MINE(Initial)

main()
# def test_func2(y, t):
#     dy_dt = np.array([2 * t])
#     return dy_dt
    
# y0 = np.array([0, 0])
# t = np.linspace(0, 10, 1000)
# sol = odeint(test_func, y0, t, args=(1, -1))

# plt.plot(sol[:, 0], sol[:, 1])
# plt.show()

# sol2 = odeint(test_func2, np.array([0]), np.linspace(0, 10, 100))
# plt.plot(np.linspace(0, 10, 100), sol2)
# plt.show()
        




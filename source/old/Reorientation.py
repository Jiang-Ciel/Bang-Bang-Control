import numpy as np
from numpy import linalg as la
from scipy import integrate as itg
from matplotlib import pyplot as plt
from numpy import matlib as mat
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from time import time
import math

InertiaTensor = np.array([[3000, 0, 0],
                          [0, 4500, 0],
                          [0, 0, 6000]])
InertiaTensor_Inv = la.inv(InertiaTensor)

Terminal_States = np.array([0.2679, 0, 0, 0, 0, 0])
Initial_States = np.zeros(6)

Direction_Sun = np.array([[-0.58, -0.08, -0.81]])
Direction_Moon = np.array([[0.40, -0.13, -0.90]])
Direction_Sensor = np.array([[0, -0.62, -0.79]])

TolerantAngle_Sun = np.radians(40)
TolerantAngle_Moon = np.radians(17)
Tolerant_Moon = np.cos(TolerantAngle_Moon)
Tolerant_Sun = np.cos(TolerantAngle_Sun)
Alpha = np.linspace(0,2*np.pi,100)

Vectors_AroundSun = np.vstack((np.cos(Alpha)*np.sin(TolerantAngle_Sun),np.sin(Alpha)*np.sin(TolerantAngle_Sun),np.cos(TolerantAngle_Sun)*np.ones(100)))
Vectors_AroundMoon = np.vstack((np.cos(Alpha)*np.sin(TolerantAngle_Moon),np.sin(Alpha)*np.sin(TolerantAngle_Moon),np.cos(TolerantAngle_Moon)*np.ones(100)))
Vector_Reference = np.array([[0, 0, 1]])

def Rotate_Matrix_Vectors2Vectors(Start_Vector_ndarray1_3, Terminal_Vector_ndarray1_3):
    """
    This function is used to calculate a Rotating_Matrix for 2 vectors
    """
    Axis_Vector = np.cross(Start_Vector_ndarray1_3, Terminal_Vector_ndarray1_3)
    Axis_Vector = Axis_Vector / la.norm(Axis_Vector)
    Theta = math.acos(np.vdot(Start_Vector_ndarray1_3, Terminal_Vector_ndarray1_3) /
                      (la.norm(Start_Vector_ndarray1_3) * la.norm(Terminal_Vector_ndarray1_3)))
    c = math.cos(Theta)
    c_1 = 1 - c
    s = math.sin(Theta)
    x = Axis_Vector[0, 0]
    y = Axis_Vector[0, 1]
    z = Axis_Vector[0, 2]
    Rotate_Matrix = np.array([[c + c_1 * x ** 2, c_1 * x * y - s * z, c_1 * x * z + s * y],
                              [c_1 * x * y + s * z, c + c_1 *
                                  y ** 2, c_1 * y * z - s * x],
                              [c_1 * z * x - s * y, c_1 * z * y + s * x, c + c_1 * z ** 2]])
    return Rotate_Matrix

Rotate_Sun = Rotate_Matrix_Vectors2Vectors(Vector_Reference, Direction_Sun)
Rotate_Moon = Rotate_Matrix_Vectors2Vectors(Vector_Reference, Direction_Moon)

Terminal_Sun = Rotate_Sun @ Vectors_AroundSun
Terminal_Moon = Rotate_Moon @ Vectors_AroundMoon




Steps = 250


def Output_Time_Slicing(Time_Stripes, t):
    Time_Stripes = Time_Stripes.reshape(3,int(Digits/3))
    # Initial_Output_States = np.array([np.sum(Time_Stripes[i,:]<0) for i in range(3)])
    Time_JudgePoints = np.array([np.sum(Time_Stripes[:, 0:i], axis=1) for i in range(1, 1+int(Digits/3))]).T
    Time_Judgement = t - Time_JudgePoints
    Output_State = np.array([(-1)**(np.sum(Time_Stripes[i,:] < 0) + np.sum(Time_Judgement[i,:] > 0)) for i in range(3)])
    return Output_State.reshape(3,)
    
    

def Kinet_Func(States, t, Time_Stripes):
    # T = 222.4080
    p1, p2, p3, w1, w2, w3 = tuple(States[i] for i in range(6))
    W = np.array([[w1, w2, w3]])
    Time_Stripes = Time_Stripes.reshape(3, int(Digits/3))
    # Initial_Output_States = np.array([np.sum(Time_Stripes[i,:]<0) for i in range(3)])
    Time_JudgePoints = np.array([np.sum(np.abs(Time_Stripes)[:, 0:i],axis=1) for i in range(1, 1 + int(Digits/3))]).T
    Time_Judgement = t - Time_JudgePoints
    Output_State = np.array([(-1)**(np.sum(Time_Stripes[i,:] < 0) + np.sum(Time_Judgement[i,:] > 0)) for i in range(3)])
    # U = 0.25*np.array([[x,y,z]])
    U = 0.25 * Output_State.reshape(1,3)
    # U = np.array([[0.25, 0, 0 ]])
    # W_dt = InertiaTensor_Inv @ (U - np.cross(W InertiaTensor @ W.T).T
    W_dt = InertiaTensor_Inv @ (U.T - np.array([[0, w3, -w2],
                                                [-w3, 0, w1],
                                                [w2, -w1, 0]]) @ InertiaTensor @ W.T)
    P = np.array([[p1, p2, p3]])
    P_dt = 0.25 * ((1 - P @ P.T) * np.identity(3) +
                    2 * np.array([[0, p3, -p2],
                                  [-p3, 0, p1],
                                  [p2, -p1, 0]]) +
                    2 * P.T @ P) @ W.T
    States_dt = np.vstack((P_dt, W_dt)).reshape(6)
    return States_dt


def Get_Rotate_Matrix(Rodrigue):
    p1, p2, p3 = tuple(Rodrigue[i] for i in range(3))
    P_ = np.array([[0, p3, -p2],
                  [-p3, 0, p1],
                  [p2, -p1, 0]])
    P_Norm = la.norm(Rodrigue) ** 2
    Rotate_Matrix = np.identity(3) + 4 * ((1 - P_Norm) / ((1 + P_Norm)** 2)) * P_ + 8 / ((1 + P_Norm)** 2) * (P_ @ P_)
    return Rotate_Matrix

def Get_Direction_Sensor(Rodrigue):
    Rotate_Matrix = Get_Rotate_Matrix(Rodrigue)
    Direction_Sensor_Temp = Rotate_Matrix.T @ Direction_Sensor.T
    return Direction_Sensor_Temp.reshape(3)



def Get_Location(Theta):
    Theta = Theta.reshape(1,3)
    Longitude = np.arcsin(Theta[0,0])
    Latitude = np.arctan2(-Theta[0,1] / np.cos(Longitude), -Theta[0,2] / np.cos(Longitude))
    Location = np.array([Longitude, Latitude])
    return np.degrees(Location).reshape(2)


def Obj_Func(Time_Stripes):
    Terminal_Time = max(np.sum(np.abs(Time_Stripes).reshape(3,int(Digits/3)),axis=1))
    t = np.linspace(0, Terminal_Time, Steps)
    Solution = itg.odeint(Kinet_Func, Initial_States, t, args=(Time_Stripes,))
    Rodrigues = Solution[:, 0:3]
    # Direction_Sun_Temp = Direction_Sun * np.ones((Steps,3))
    # Direction_Moon_Temp = Direction_Moon * np.ones((Steps, 3))
    Direction_Sensor_Temp = np.array(list(map(Get_Direction_Sensor, Rodrigues)))
    # Location = np.array(list(map(Get_Location, Sensors)))
    # Direction_Sensor = Direction_Sensor.T * np.ones((Steps,3))
    Cone_Moon = np.sum(Direction_Moon * Direction_Sensor_Temp, axis=1)
    Cone_Sun = np.sum(Direction_Sun * Direction_Sensor_Temp, axis=1)
    # print(Cone_Moon.shape,Cone_Sun.shape)
    # print(np.sum(Cone_Moon > np.cos(TolerantAngle_Moon)) , np.sum(Cone_Sun > np.cos(TolerantAngle_Sun)))
    Cone_Constraint = 10 *  (np.sum(Cone_Moon > Tolerant_Moon) + np.sum(Cone_Sun > Tolerant_Sun))
    
    Terminal_State_Constraint = 5000 * la.norm(Solution[-1,:] - Terminal_States)

    # print("Time Used:",Time_Used)
    # print("Terminal_State_Cons:",Terminal_State_Constraint)
    # print("Cone_Constr:",Cone_Constraint)
    return Terminal_Time + Terminal_State_Constraint + Cone_Constraint 

def Obj_Func_Print(Time_Stripes):
    Terminal_Time = max(np.sum(np.abs(Time_Stripes).reshape(3,int(Digits/3)),axis=1))
    t = np.linspace(0, Terminal_Time, Steps)
    Solution = itg.odeint(Kinet_Func, Initial_States, t, args=(Time_Stripes,))
    Rodrigues = Solution[:, 0:3]
    Direction_Sun_Temp = Direction_Sun * np.ones((Steps,3))
    Direction_Moon_Temp = Direction_Moon * np.ones((Steps, 3))
    Direction_Sensor_Temp = np.array(list(map(Get_Direction_Sensor, Rodrigues)))
    # Location = np.array(list(map(Get_Location, Sensors)))
    # Direction_Sensor = Direction_Sensor.T * np.ones((Steps,3))
    Cone_Moon = np.sum(Direction_Moon_Temp * Direction_Sensor_Temp, axis=1)
    Cone_Sun = np.sum(Direction_Sun_Temp * Direction_Sensor_Temp, axis=1)
    # print(Cone_Moon.shape,Cone_Sun.shape)
    # print(np.sum(Cone_Moon > np.cos(TolerantAngle_Moon)) , np.sum(Cone_Sun > np.cos(TolerantAngle_Sun)))
    Cone_Constraint = (np.sum(Cone_Moon > Tolerant_Moon) + np.sum(Cone_Sun > Tolerant_Sun))
    
    Terminal_State_Constraint = 5000 * la.norm(Solution[-1,:] - Terminal_States)

    print("Time Used:",Terminal_Time)
    print("Terminal_State_Cons:",Terminal_State_Constraint)
    print("Cone_Constr:",Cone_Constraint)
    return None




Digits = 21
MemNum = 45
Cycle = 5
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


# def Test_Func(Start: np.ndarray):
#     a, b = 100, 50
#     alpha = 1.5
#     x0 = mat.repmat(Start, MemNum, 1)
#     Dir = a * Rns.rand(MemNum, Digits) - b
#     x0 += Dir
#     Hbestv = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
#     # print(Hbestv.shape)
#     Hbest = x0.copy()
#     Radius0 = np.array([np.std(Hbest - x0[i,:], axis=0) for i in range(MemNum)])
#     # Radius0 = mat.repmat(np.std(Hbest, axis=0), MemNum, 1)
#     Pace = 3 * Rns.rand(MemNum, Digits) * Radius0
#     for i in range(Cycle):
#         Hbestv_Normal = Hbestv/np.sum(Hbestv)
#         EXP_Hbestv = np.exp(Hbestv_Normal)
#         SoftMax = np.sum(EXP_Hbestv) / (EXP_Hbestv + 1e-15)
#         SoftMax = SoftMax / np.max(SoftMax)
#         Dir = np.array([np.mean(SoftMax * (Hbest - x0[i,:]), axis=0) for i in range(MemNum)])
#         Radius1 = np.array([np.std(Hbest - x0[i,:], axis=0) for i in range(MemNum)])
#         Bias = 0.8 / (1 + np.exp(0.5 * Radius1 / (Radius0 + 1e-15)-1))
#         Trust=alpha * Rns.rand(MemNum, Digits) - Bias
#         Trust=np.where(Trust > 0, 1, -1)
#         Pace= 1.5 * Trust * Dir / (1 + np.exp(Pace / (Dir + 1e-15)-1))
#         Radius0=Radius1.copy()
#         x0 += Pace
#         Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
#         Hbest=np.where(Check < Hbestv, x0, Hbest)
#         Hbestv=np.where(Check < Hbestv, Check, Hbestv)
#         print("MINE", i + 1)
#         print("--------------------")
#         print("radius:", Radius1.max(),Radius1.min())
#         print("Pace:", np.abs(Pace).max(),np.abs(Pace).min())
#         print("--------------------")
#         print("Obj.Func:", np.min(Hbestv))
#         print("Componets MAX:", np.max(np.abs(Hbest[np.argmin(Hbestv),:])))
#         print("Componets MIN:", np.min(np.abs(Hbest[np.argmin(Hbestv),:])))
#         print('====================')
#         print("")
#     return Hbestv.min(),Hbest[np.argmin(Hbestv),:]
        
        

    


def MINE(Start: np.ndarray):
    a, b = 100, 50
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
    H = []
    radius0 = la.norm(np.std(Hbest, axis=0))
    Pace = 3 * radius0
    P = []
    R = []
    Po = []
    Ratio = []
    Bi = []
    Direction = np.zeros((MemNum, Digits))
    APace = alpha * (1 + np.exp(alpha - 1))
    Size = MemNum * Digits
    for i in range(Cycle):
        Rns = np.random.RandomState(np.random.randint(0,100,1))
        radius1 = np.std(Hbest, axis=0)
        po = np.mean(Hbest, axis=0)
        distance = Hbest - po
        distance = la.norm(distance, axis=1).reshape(MemNum, 1)
        Po.append(po)
        radius1 = la.norm(radius1)
        Judge = 1 / (1 + np.exp(Pace / (radius1 + 1e-15) - 1))
        # Trust = max[1 - Judge / 3, 0]
        Trust = 1 / (1 + np.exp(1.5 * Judge - 1))
        # Trust = 1 / (1 + np.exp(np.sum(distance / radius1 * np.log(distance / radius1)) - 1))
        # print("ratio:", radius1 / radius0)
        # print("Trust:",Trust)
        # print("Judge:",Judge)
        Pace = APace * radius1 * Judge
        # Pace = alpha * (1 + np.exp(alpha - 1)) * radius1 * Judge
        Bias = alpha * Judge
        radius0 = radius1
        if i % 7 == 0 or i % 7 == 2 or i % 7 == 4:
            Ratio.append(Pace / radius0)
            Bi.append(Bias)
            P.append(Pace)
            R.append(radius0)
            Sort = np.argsort(Hbestv, axis=0)
            Chaos = 1 * np.random.random(Digits) + 0.5
            Chaos.reshape(1,Digits)
            Ref = Hbest[Sort[:,0],:]
            Hbelief = (2 * Rns.rand(MemNum, Digits) - Bias * Chaos).tolist()
            Direction = np.array(list(map(Get_Direction, [Ref for _ in range(MemNum)], x0.tolist(), Hbelief)))
            Direction = Direction / (la.norm(Direction,axis=1).reshape(MemNum,1))
            Gbelief = Rns.normal(Trust,min([1-Trust,Trust])/3, (MemNum,Digits))
            Dir =np.abs(1 - Gbelief)  * Dir + Gbelief * Direction
            if Dir.dtype != "float64":
                Dir = Dir.astype("float")
            Dir = Dir / (la.norm(Dir,axis=1).reshape(MemNum,1))
            x0 = x0 + Pace * Dir
            Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
            Hbest=np.where(Check < Hbestv, x0, Hbest)
            Hbestv=np.where(Check < Hbestv, Check, Hbestv)
        elif i % 7 == 3 :
            x1 = Hbest.copy()
            x1 = x1.reshape(Size,)
            Rns.shuffle(x1)
            x1 = x1.reshape(MemNum, Digits)
            Check = np.array(list(map(Obj_Func, x1))).reshape(MemNum, 1)
            Hbest=np.where(Check < Hbestv, x1, Hbest)
            Hbestv = np.where(Check < Hbestv, Check, Hbestv)
        
        elif i % 7 == 5:
            x0 = x0.reshape(Size,)
            Rns.shuffle(x0)
            x0 = x0.reshape(MemNum, Digits)
            Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
            Hbest=np.where(Check < Hbestv, x0, Hbest)
            Hbestv = np.where(Check < Hbestv, Check, Hbestv)

        elif i % 7 == 1:
            x1 = Hbest.copy()
            x1 = x1.reshape(Size,)
            Rns.shuffle(x1)
            Index = Rns.randint(0, Size, int(Size / 3))
            x1[Index] = 0
            x1 = x1.reshape(MemNum, Digits)
            Check = np.array(list(map(Obj_Func, x1))).reshape(MemNum, 1)
            Hbest=np.where(Check < Hbestv, x1, Hbest)
            Hbestv = np.where(Check < Hbestv, Check, Hbestv)
        else:
            x0 = x0.reshape(Size,)
            Rns.shuffle(x0)
            Index = Rns.randint(0, Size, int(Size / 3))
            x0[Index] = 0
            x0 = x0.reshape(MemNum, Digits)
            Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
            Hbest=np.where(Check < Hbestv, x0, Hbest)
            Hbestv = np.where(Check < Hbestv, Check, Hbestv)


        Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
        Hbest=np.where(Check < Hbestv, x0, Hbest)
        Hbestv=np.where(Check < Hbestv, Check, Hbestv)
        # Place = np.where(Check - Hbestv < 0)
        # Hbestv[Place,:] = Check[Place,:]
        # Hbest[Place,:] = x0[Place,:]
        h = Hbest[int(np.argmin(Hbestv)),:].tolist()
        print("MINE", i + 1, " ", i % 7)
        print("--------------------")
        print("radius:", radius1)
        print("Pace:", Pace)
        print("--------------------")
        print("Obj.Func:", np.min(Hbestv))
        print("Obj.Func--:", np.min(Check))
        print("Componets MAX:", np.max(np.abs(Hbest[int(np.argmin(Hbestv)),:])))
        print("Componets MIN:", np.min(np.abs(Hbest[int(np.argmin(Hbestv)),:])))
        print("X MAX:", np.max(np.abs(x0)))
        print("X MIN:", np.min(np.abs(x0)))
        print('====================')
        print("")
        Data = pd.DataFrame(Hbest[np.argmin(Hbestv)])
        Data.to_excel("Data.xlsx",sheet_name="Data")
        h.append(np.min(Hbestv))
        H.append(h)
    Po = np.array(Po)
    # f,(AX1,AX2,AX3)=plt.subplots(3)
    # # plt.figure("MINE：Time-Position")
    # for i in range(Digits):
    #     AX3.plot(np.arange(Cycle),Po[:,i])
    # plt.xlabel("Times")  
    # plt.ylabel("Components") 
    # # plt.figure("MINE：Time-Radius/Pace")
    # Radius,=AX1.plot(range(Cycle), R, label="Radius")
    # Pace,=AX1.plot(range(Cycle), P,linestyle=":", label="Pace")
    # AX1.legend(loc="upper right")
    # plt.xlabel("Times")  
    # plt.ylabel("Value")
    # # plt.figure("MINE：Time-Radius/Pace,Bias")
    # ratio,=AX2.plot(range(Cycle), Ratio, linestyle=":",label="Pace/Radius")
    # Bias,=AX2.plot(range(Cycle), Bi, label="Bias")
    # AX2.legend(loc="upper right")
    # plt.xlabel("Steps",fontdict=font)  
    # plt.ylabel("Value of 'Pace/Radius' and 'Bias'",fontdict=font)
    H = np.array(H)
    value = H[:,Digits]
    value.reshape(1,Cycle)
    return value,Hbestv.min(),Hbest[np.argmin(Hbestv),:]


def PSO(Start:np.ndarray):
    GroupNum = 45
    # Digits = 60
    Coefficient = mat.repmat(Start,GroupNum,1) + 200* np.random.rand(GroupNum, Digits) - 100
    v0 = 400 * np.random.rand(GroupNum, Digits) - 200
    # Cycle = 250
    pbest = Coefficient.view()
    pbestv = np.zeros((GroupNum, 1))
    pbestv1 = np.zeros((GroupNum, 1))
    H = []
    for Inner in range(GroupNum):
        Check = Obj_Func(Coefficient[Inner,:])
        pbestv[Inner,:] = Check
    gbest = mat.repmat(Coefficient[np.argmin(pbestv),:], GroupNum, 1)
    wmax, wmin, c1, c2 = 0.9, 0.4, 2, 2
    R = []
    Po = []
    P = []
    for i in range(Cycle):
        w = wmax - i * (wmax - wmin) / Cycle
        r1 = np.random.rand(GroupNum, 1)
        r2 = np.random.rand(GroupNum, 1)
        v1 = w * v0 + c1 * r1 * (pbest - Coefficient) + c2 * r2 * (gbest - Coefficient)
        Coefficient = Coefficient + v1
        pace = la.norm(v1,axis=1)
        pace = np.max(pace)
        P.append(pace)
        Coefficient = Coefficient + v1
        radius = np.std(pbest, axis=0)
        radius = la.norm(radius)
        R.append(radius)
        po = np.mean(pbest, axis=0)
        Po.append(po)
        for Inner in range(GroupNum):
            Check = Obj_Func(Coefficient[Inner,:])
            pbestv1[Inner,:] = Check 
        for j in range(GroupNum):
            if pbestv1[j,:] < pbestv[j,:]:
                pbest[j,:] = Coefficient[j,:]
                pbestv[j,:] = pbestv1[j,:]
        v0 = v1.view()
        gbest = mat.repmat(pbest[np.argmin(pbestv),:], GroupNum, 1)
        h = pbest[np.argmin(pbestv),:].tolist()
        print("PSO", i)
        print("--------------------")
        print("Obj.Func:", np.min(pbestv))
        print("Components MAX:", np.max(np.abs(pbest[np.argmin(pbestv),:])))
        print("Components MIN:", np.min(np.abs(pbest[np.argmin(pbestv),:])))
        print('====================')
        h.append(pbestv.min())
        H.append(h)
    H = np.array(H)
    # out = pd.DataFrame(H)
    # out.to_excel("Record.xlsx",sheet_name="PSO")
    value = H[:, Digits]
    value.reshape(1, Cycle)
    Po = np.array(Po)
    R = np.array(R)
    # plt.figure("PSO：Time-Position")
    # for i in range(Digits):
    #     plt.plot(np.arange(Cycle), Po[:, i])
    # plt.xlabel("Times")  
    # plt.ylabel("Components") 
    # plt.figure("PSO：Time-Radius/Pace")
    # Pace,=plt.plot(range(Cycle), P, label="Pace")
    # Radius,=plt.plot(range(Cycle), R, label="Radius")
    # plt.legend(loc="upper right")
    # plt.xlabel("Times")  
    # plt.ylabel("Value")
    return value,pbestv.min(),pbest[np.argmin(pbestv),:]


def main():
    Time_Slicing = 100 * np.random.rand(21) - 50
    # Time_Slicing = 250 * np.array([0.5, 0.5, 0, 0, 0, 0, 0, -0.2, 0.53, 0.27, 0, 0, 0, 0, 0.075, 0.4-0.075, 0.45, 0.15, 0, 0, 0])
    value, Hbestv, Hbest = MINE(Time_Slicing)
    # value, Hbestv, Hbest = (Time_Slicing)
    # value, Hbestv, Hbest = PSO(Time_Slicing)
    # print(value, Hbestv, Hbest)
    Terminal_Time = max(np.sum(np.abs(Hbest).reshape(3,int(Digits/3)),axis=1))
    Time = np.linspace(0, Terminal_Time, Steps)
    # print(Terminal_Time)
    Obj_Func_Print(Hbest)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    Solution = itg.odeint(Kinet_Func, Initial_States, Time, args=(Hbest,))
    Rodrigues = Solution[:, 0:3]
    # print(Rodrigues.shape)
    Direction_Sensor_Temp = np.array(list(map(Get_Direction_Sensor, Rodrigues)))
    # print(Direction_Sensor.shape)
    Location = np.array(list(map(Get_Location, Direction_Sensor_Temp)))
    Location_Sun = np.array(list(map(Get_Location, Terminal_Sun.T)))
    Location_Moon = np.array(list(map(Get_Location, Terminal_Moon.T)))
    
    ax1.plot(Location[:, 0], Location[:,1])
    ax1.plot(Location_Moon[:, 0], Location_Moon[:,1])
    ax1.plot(Location_Sun[:, 0], Location_Sun[:, 1])
    ax1.grid()
    plt.xlim([-80, 60])
    plt.ylim([-40, 60])
    ax2.plot(Time, Solution[:, 0], Time, Solution[:, 1], Time, Solution[:, 2])
    ax2.grid()
    x, = ax3.plot(Time, Solution[:, 3], label="x")
    y, = ax3.plot(Time, Solution[:, 4], label="y")
    z, = ax3.plot(Time, Solution[:, 5], label="z")
    ax3.legend(loc="upper right")
    ax3.grid()
    Output = []
    for t in Time:
        Output.append(Output_Time_Slicing(Hbest, t))
    Output = np.array(Output).T
    ax4.plot(Time,Output[0,:],label="U1")
    ax4.plot(Time,Output[1,:],label="U2")
    ax4.plot(Time,Output[2,:],label="U3")
    ax4.legend(loc="upper right")
    plt.ylim([-1.5, 1.5])
    plt.xlim([-5,Terminal_Time])
    ax4.grid()
    plt.show()

# def test():
#     Time_Slicing = 250 * np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.2, 0.53, 0, 0, 0, 0, 0, 0, 0, 0, 0.075, 0.4-0.075, 0.45, 0, 0, 0, 0, 0, 0, 0])
#     Value= Obj_Func(Time_Slicing)
#     print(Value)
main()
# test()
# value,Hbestv,Hbest =PSO(Time_Slicing)

# print("value:",value)
# print("Hbestv:",Hbestv)
# print("Hbest:",Hbest)
# Sensors = []
# for i in range(sol.shape[0]):
#     P = sol[i:i + 1, 0:3]
#     P = P.reshape(3)
#     Rotate_Matrix = Get_Rotate_Matrix(P)
#     Sensor = Rotate_Matrix.T @ Direction_Sensor.T
#     Sensors.append(Sensor.T)
# print(Sensors[0])
# Location = np.array(list(map(Get_Location, Sensors)))
# print(Location.shape)
# plt.figure("1")
# plt.plot(Location[:, 0], Location[:,1])
# plt.grid()
# plt.xlim([-80, 60])
# plt.ylim([-40,60])
# plt.figure("2")
# plt.plot(t,sol[:,0],t,sol[:,1],t,sol[:,2])
# plt.grid()
# # plt.plot(t, Location[:,1],t,Location[:,0])
# plt.figure("3")
# plt.plot(t, sol[:, 3], label="x")
# plt.plot(t, sol[:, 4], label="y")
# plt.plot(t, sol[:, 5], label="z")
# plt.legend()
# plt.grid()
# plt.show()

    
                                  
    


                          
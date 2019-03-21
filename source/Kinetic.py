import numpy as np
from numpy import linalg as la
from numpy import matlib as mat
from scipy import integrate as itg
from matplotlib import pyplot as plt
import pandas as pd
from time import time
import math


class Kinetic():

    """
    Description
    ----------
    This class is a bunch of functions and parameters related to the kinetic model of a rigid body system
    and the Bang-Bang-Control principles for convinience of simulations.

    Parameters
    ----------
    Digits : int
        This parameter is the digits of input target for optimizations 
        in the Bang-Bang-Control.
    Steps : int
        This parameter is the amount of steps for the ode function.
    
    Examples
    ----------
    >>> import numpy as np
    >>> from Kinetic import Kinetic
    >>> Test = Kinetic(21)
    >>> Test
    <Kinetic.Kinetic at 0x2044e0d2ba8>
    >>> Test.__class__
    Kinetic.Kinetic
    >>> Test.Digits
    21
    >>> Test2 = Kinetic(21, 20)
    >>> Test.Step
    20
    """

    def __init__(self, Digits, Step = 0.8):
        self.InertialTensor = np.array([[3000, 0, 0],
                                        [0, 4500, 0],
                                        [0, 0, 6000]])

        self.InertialTensor_Inv = la.inv(self.InertialTensor)
        self.Thrust = 0.25
        self.Digits = Digits
        self.Step = Step
        

        self.Terminal_States = np.array([0.2679, 0, 0, 0, 0, 0])
        self.Initial_States = np.zeros(6)

        self.Direction_Sun = np.array([[-0.58, -0.08, -0.81]])
        self.Direction_Moon = np.array([[0.40, -0.13, -0.90]])
        self.Direction_Sensor = np.array([[0, -0.62, -0.79]])

        self.TolerantAngle_Sun = math.radians(40)
        self.TolerantAngle_Moon = math.radians(17)
        self.Tolerant_Moon = np.cos(self.TolerantAngle_Moon)
        self.Tolerant_Sun = np.cos(self.TolerantAngle_Sun)
        self.Alpha = np.linspace(0,2*np.pi,100)

        self.Vectors_AroundSun = np.vstack((np.cos(self.Alpha) * np.sin(self.TolerantAngle_Sun),
                                            np.sin(self.Alpha) * np.sin(self.TolerantAngle_Sun),
                                            self.Tolerant_Sun * np.ones(100)))
        self.Vectors_AroundMoon = np.vstack((np.cos(self.Alpha) * np.sin(self.TolerantAngle_Moon),
                                             np.sin(self.Alpha)*np.sin(self.TolerantAngle_Moon),
                                             self.Tolerant_Moon * np.ones(100)))
        self.Vector_Reference = np.array([[0, 0, 1]])
        self.Rotate_Sun = self.RotMat_Vec2Vec(self.Vector_Reference, self.Direction_Sun)
        self.Rotate_Moon = self.RotMat_Vec2Vec(self.Vector_Reference, self.Direction_Moon)

        self.Terminal_Sun = self.Rotate_Sun @ self.Vectors_AroundSun
        self.Terminal_Moon = self.Rotate_Moon @ self.Vectors_AroundMoon  

    @staticmethod
    def RotMat_Vec2Vec(StartVector, TerminalVector):
       
        """
        Description
        ----------

        This function is used to calculate a Rotate Matrix from StartVector to TerminalVector
        in order to transfer a series of vectors which share the same rotation from StartVector 
        to their termination.

        Parameters
        ----------
        StartVector : numpy_array, shape(1, 3)
            Vector represent the initial pointing vector of the rotation movement.
        TerminalVector : numpy_array, shape(1, 3)
            Vector represent the terminal pointing vector of the rotation movement.
        
        Returns
        ----------
        Rotate_Matrix : numpy_array, shape(3, 3)
            A symmtric matrix stand for the rotation movement from StartVector to 
            Terminal vector
        
        Examples
        ----------
        >>> import numpy as np
        >>> from Kinetic import Kinetic
        >>> StartVector = np.array([[0, 1, 0]])
        >>> TerminalVector = np.array([[1, 0, 0]])
        >>> Kinetic.RotMat_Vec2Vec
        <function Kinetic.RotMat_Vec2Vec at 0x00000134BE5AA378>
        >>> Kineitc.RotMat_Vec2Vec(StartVector, TerminalVector)
        array([[ 6.123234e-17,  1.000000e+00,  0.000000e+00],
               [-1.000000e+00,  6.123234e-17, -0.000000e+00],
               [-0.000000e+00,  0.000000e+00,  1.000000e+00]])
        """
        
        Axis_Vector = np.cross(StartVector, TerminalVector)
        Axis_Vector = Axis_Vector / la.norm(Axis_Vector)
        Theta = math.acos(np.vdot(StartVector, TerminalVector) /
                        (la.norm(StartVector) * la.norm(TerminalVector)))
        c = math.cos(Theta)
        c_1 = 1 - c
        s = math.sin(Theta)
        x = Axis_Vector[0, 0]
        y = Axis_Vector[0, 1]
        z = Axis_Vector[0, 2]
        Rotate_Matrix = np.array([[c + c_1 * x ** 2, c_1 * x * y - s * z, c_1 * x * z + s * y],
                                [c_1 * x * y + s * z, c + c_1 *y ** 2, c_1 * y * z - s * x],
                                [c_1 * z * x - s * y, c_1 * z * y + s * x, c + c_1 * z ** 2]])
        return Rotate_Matrix
    
    @staticmethod
    def RotMat_Rod(Rodrigue):
        
        """
        Description
        ----------
        This function is used to transfer Rodrigues to a Matrix for further use. 

        Parameters
        ----------
        Rodrigue : numpy_array, shape(3,)
            Vector represents the Rodrigues for a specific time point.

        Returns
        ----------
        Rotate_Matrix : numpy_array, shape(3, 3)
            2-D symmetry Matrix represent the Rotate Matrix

        Examples
        ----------
        >>> from Kinetic import Kinetic
        >>> import numpy as np
        >>> Rodrigue = np.array([0, 1, 0.5])
        >>> Kinetic.RotMat_Rod(Rodrigue)
        array([[-0.97530864, -0.09876543,  0.19753086],
               [ 0.09876543,  0.60493827,  0.79012346],
               [-0.19753086,  0.79012346, -0.58024691]])
        """
        
        p1, p2, p3 = tuple(Rodrigue[i] for i in range(3))
        P_ = np.array([[0, p3, -p2],
                       [-p3, 0, p1],
                       [p2, -p1, 0]])
        P_Norm = la.norm(Rodrigue) ** 2
        Rotate_Matrix = np.identity(3) + 4 * ((1 - P_Norm) / ((1 + P_Norm)** 2)) * P_ + 8 / ((1 + P_Norm)** 2) * (P_ @ P_)
        return Rotate_Matrix


    def __Kinet_Func(self, States, t, Time_Stripes):

        """
        Description
        ----------
        This function is vital in computing the components in Object function for the final optimization problem,
        this function is a fully defination of the ODE describing the kinetic state propragation,which is merely 
        
        Parameters
        ----------
        States : numpy_array, shape(6,)
            A 6-D vector whose first 3 dimensions stand for the Rodrigues followed 
            with 3 dimensions represent angular vectors in the components in X,Y,Z
            axis of the fix-body-coordinate, respectively
        t : float
            Time index whose unit is "(s) second"
        Time_Stripes : numpy_array, shape(self.Digits,)
            A vector stand for the switch time points series which could be divided into 3 groups that stand for 
            that in relation with X, Y, Z axis, respectively. 
        Returns
        ----------
        States_dt : numpy_array, shape(6,)
            States' differentiation w.r.t time. 
        """

        p1, p2, p3, w1, w2, w3 = tuple(States[i] for i in range(6))
        P = np.array([States[0:3]])
        W = np.array([States[3:6]])
        Time_Stripes = Time_Stripes.reshape(3, int(self.Digits/3))
        Time_JudgePoints = np.array([np.sum(np.abs(Time_Stripes)[:, 0:i],axis=1) for i in range(1, 1 + int(self.Digits/3))]).T
        Time_Judgement = t - Time_JudgePoints
        Output_State = np.array([(-1)**(np.sum(Time_Stripes[i,:] < 0) + np.sum(Time_Judgement[i,:] > 0)) for i in range(3)])
        U = self.Thrust * Output_State.reshape(1,3)
        W_dt = self.InertialTensor_Inv @ (U.T -
                                        np.array([[0, w3, -w2],
                                                  [-w3, 0, w1],
                                                  [w2, -w1, 0]])
                                        @ self.InertialTensor @ W.T)
        P_dt = 0.25 * ((1 - P @ P.T) * np.identity(3) +
                        2 * np.array([[0, p3, -p2],
                                      [-p3, 0, p1],
                                      [p2, -p1, 0]]) +
                        2 * P.T @ P) @ W.T
        States_dt = np.vstack((P_dt, W_dt)).reshape(6)
        return States_dt


    def __Direction_Sensor(self, Rodrigue):
        """

        """
        Direction_Sensor_Temp = self.RotMat_Rod(Rodrigue).T @ self.Direction_Sensor.T
        return Direction_Sensor_Temp.reshape(3)

    @staticmethod
    def LocVec_Pol(LocVec_Per):

        """
        Description
        ----------
        This function is used to tranfer the pointing vectors for sensors in a perpendicular 
        Body-Fix-Coordinate to that of Polar Coordinate system representation in convinience 
        to make the illusion of maneuver path and the Keep-Out Cones constraints obvious.

        Parameters
        ----------
        LocVec_Per : numpy_array
            A 3-D vector whose basis are perpendicular.
        
        Returns
        ----------
        LocVec_Polar : numpy_array, shape(2,)
            A 2-D vector corresponding to the Body-Fixed-Polar-Coordinate-System.
        
        Examples
        ----------
        >>> from Kinetic import Kinetic
        >>> import numpy as np
        >>> LocVec_Per = np.array([200, 500, 300])
        >>> Kinetic.LocVec_Pol(LocVec_Per)
        array([  18.93182318, -120.96375653])
        """

        LocVec_Per = LocVec_Per / la.norm(LocVec_Per)
        Longitude = math.asin(LocVec_Per[0])
        Latitude = np.arctan2(-LocVec_Per[1] / math.cos(Longitude), -LocVec_Per[2] / math.cos(Longitude))
        LocVec_Polar = np.degrees(np.array([Longitude, Latitude])).reshape(2)
        return LocVec_Polar

    
    def Obj_Func(self, Time_Stripes):
        
        """
        Description
        ----------
        This function is the objective function for the time-optimal maneuver problem.

        Parameters
        ----------
        Time_Stripes : numpy_array
            A 'self.Digits' dimension vector stands for the switch point.

        Returns
        ----------
        Obj_Result : float
            The value of the function stems from: 1)Terminal state error
            2)Cone Constraints 3)Time

        Example
        ----------
        >>> from Kinetic import Kinetic
        >>> import numpy as np
        >>> test = Kinetic(21)
        >>> Swiches = np.ones(21)
        >>> test.Obj_Func(Switches)
        1343.9545826260917
        """

        if type(Time_Stripes) is list:
            Time_Stripes = np.array(Time_Stripes)
        Terminal_Time = max(np.sum(np.abs(Time_Stripes).reshape(3,int(self.Digits/3)),axis=1))
        Steps = int(Terminal_Time / self.Step)
        t = np.linspace(0, Terminal_Time, Steps)
        Solution = itg.odeint(self.__Kinet_Func, self.Initial_States, t, args=(Time_Stripes,))
        Rodrigues = Solution[:, 0:3]
        Direction_Sun_Temp = self.Direction_Sun * np.ones((Steps,3))
        Direction_Moon_Temp = self.Direction_Moon * np.ones((Steps, 3))
        Direction_Sensor_Temp = np.array(list(map(self.__Direction_Sensor, Rodrigues)))

        Cone_Moon = np.sum(Direction_Moon_Temp * Direction_Sensor_Temp, axis=1)
        Cone_Sun = np.sum(Direction_Sun_Temp * Direction_Sensor_Temp, axis=1)
        Cone_Constraint = 5000 * (np.sum(Cone_Moon > self.Tolerant_Moon) + np.sum(Cone_Sun > self.Tolerant_Sun))

        Terminal_State_Constraint = 5000 * la.norm(Solution[-1,:] - self.Terminal_States)
        Obj_Result = Terminal_State_Constraint + Cone_Constraint + Terminal_Time
        return Obj_Result

    # def test(self):
    #     Steps = 500
    #     Time_Stripes = np.random.rand(21)
    #     Terminal_Time = max(np.sum(np.abs(Time_Stripes).reshape(3,int(self.Digits/3)),axis=1))
    #     t = np.linspace(0, Terminal_Time, Steps)
    #     Solution = itg.odeint(self.__Kinet_Func, self.Initial_States, t, args=(Time_Stripes,))
    #     print(Solution.shape)
    #     print(self.Vectors_AroundMoon.shape)
    #     print(self.Vectors_AroundSun.shape)
    
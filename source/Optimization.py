import numpy as np
from multiprocessing import Pool
import multiprocessing
from numpy import linalg as la
from numpy import matlib as mat
from scipy import integrate as itg
from matplotlib import pyplot as plt
import pandas as pd
import math

class Optimization():
    
    """
    Description
    ----------
    This class is a bunch of functions and parameters related to the optimization functions.

    Parameters
    ----------
    Digits : int
        This parameter is the digits of the dimension of the Inputs
    
    Examples
    ----------
    >>> import numpy as np
    >>> from Optimization import Optimization as Opt
    >>> Test = Opt(21)
    >>> Test
    <Optimization.Optimization object at 0x00000210C1E23CF8>
    >>> Test.__class__
    <class 'Optimization.Optimization'>
    """

    def __init__(self, Digits):
        self.Digits = Digits


    def __Get_Direction(self, Ref, x0, Hbelief, MemNum):
        base = Ref - np.array(x0)
        Normal = la.norm(base, axis=1).reshape(MemNum,1)
        det = Normal.argmin()
        base = np.delete(base, det, axis=0)
        Normal = np.delete(Normal, det, axis=0)
        base = base[0:self.Digits,:]
        Normal = Normal[0:self.Digits,:]
        if base.dtype != "float":
            base = base.astype("float")
        base = base / Normal
        return np.dot(np.array(Hbelief), base).reshape(self.Digits,)


    def MINE(self, Obj_Func, Start, MemNum, Cycle):
        
        """
        Description
        ----------
        This function is a metaheuristic algorithm for optimization.

        Parameters
        ----------
        Obj_Func: function
            Objective function.
        Start : numpy_array
            This parameter is a initial point for the following optimization operation.
        MemNum : int
            Number of points in searching for every step.
        Cycle : int
            Round of circulation.

        Returns
        ----------
        Obj_Value : float
            The value of the Objective function.
        Opt_Result : numpy_array
            The result of the optimization.

        Examples
        ----------
        >>> import numpy as np
        >>> from Optimization import Optimization as Opt
        >>> import numpy as np
        >>> 
        """
        
        a, b = 100, 50
        alpha = 1.5
        x0 = mat.repmat(Start, MemNum, 1)
        Dir = a * np.random.rand(MemNum, self.Digits) - b
        x0 += Dir 
        Dir = Dir / Dir.min()
        if Dir.dtype != "float64":
            Dir = Dir.astype("float")
        Dir = Dir / (la.norm(Dir, axis=1).reshape(MemNum, 1))
        Hbestv = np.array(list(map(Obj_Func, x0))).reshape(MemNum,1)
        Hbest = x0.copy()
        radius0 = la.norm(np.std(Hbest, axis=0))
        Pace = 3 * radius0
        Direction = np.zeros((MemNum, self.Digits))
        APace = alpha * (1 + np.exp(alpha - 1))
        Size = MemNum * self.Digits
        Cores = multiprocessing.cpu_count()
        for i in range(Cycle):
            Rns = np.random.RandomState(np.random.randint(0,100,1))
            radius1 = np.std(Hbest, axis=0)
            po = np.mean(Hbest, axis=0)
            distance = Hbest - po
            distance = la.norm(distance, axis=1).reshape(MemNum, 1)
            radius1 = la.norm(radius1)
            Judge = 1 / (1 + np.exp(Pace / (radius1 + 1e-15) - 1))
            Trust = 1 / (1 + np.exp(1.5 * Judge - 1))
            Pace = APace * radius1 * Judge
            Bias = alpha * Judge
            radius0 = radius1
            Sort = np.argsort(Hbestv, axis=0)
            Chaos = 1 * np.random.random(self.Digits) + 0.5
            Chaos.reshape(1,self.Digits)
            Ref = Hbest[Sort[:,0],:]
            Hbelief = (2 * Rns.rand(MemNum, self.Digits) - Bias * Chaos).tolist()
            Direction = np.array(list(map(self.__Get_Direction, [Ref for _ in range(MemNum)], x0.tolist(), Hbelief,MemNum * [MemNum,])))
            Direction = Direction / (la.norm(Direction,axis=1).reshape(MemNum,1))
            Gbelief = Rns.normal(Trust,min([1-Trust,Trust])/3, (MemNum,self.Digits))
            Dir =np.abs(1 - Gbelief)  * Dir + Gbelief * Direction
            if Dir.dtype != "float64":
                Dir = Dir.astype("float")
            Dir = Dir / (la.norm(Dir,axis=1).reshape(MemNum,1))
            x0 = x0 + Pace * Dir
            with Pool(Cores) as p:
                Check = np.array(p.map(Obj_Func, x0)).reshape(MemNum, 1)
                p.close()
                p.join()
            # if i % 7 == 0 or i % 7 == 2 or i % 7 == 4:
            #     Sort = np.argsort(Hbestv, axis=0)
            #     Chaos = 1 * np.random.random(self.Digits) + 0.5
            #     Chaos.reshape(1,self.Digits)
            #     Ref = Hbest[Sort[:,0],:]
            #     Hbelief = (2 * Rns.rand(MemNum, self.Digits) - Bias * Chaos).tolist()
            #     Direction = np.array(list(map(self.__Get_Direction, [Ref for _ in range(MemNum)], x0.tolist(), Hbelief,MemNum * [MemNum,])))
            #     Direction = Direction / (la.norm(Direction,axis=1).reshape(MemNum,1))
                # Gbelief = Rns.normal(Trust,min([1-Trust,Trust])/3, (MemNum,self.Digits))
                # Dir =np.abs(1 - Gbelief)  * Dir + Gbelief * Direction
                # if Dir.dtype != "float64":
                    # Dir = Dir.astype("float")
                # Dir = Dir / (la.norm(Dir,axis=1).reshape(MemNum,1))
                # x0 = x0 + Pace * Dir
                # with Pool(Cores) as p:
                    # Check = np.array(p.map(Obj_Func, x0)).reshape(MemNum, 1)
                    # p.close()
                    # p.join()
                # Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
                # Hbest=np.where(Check < Hbestv, x0, Hbest)
                # Hbestv=np.where(Check < Hbestv, Check, Hbestv)
            # elif i % 7 == 3 :
                # x1 = Hbest.copy()
                # x1 = x1.reshape(Size,)
                # Rns.shuffle(x1)
                # x1 = x1.reshape(MemNum, self.Digits)
                # with Pool(Cores) as p:
                    # Check = np.array(p.map(Obj_Func, x0)).reshape(MemNum, 1)
                    # p.close()
                    # p.join()
                # Check = np.array(list(map(Obj_Func, x1))).reshape(MemNum, 1)
                # Hbest=np.where(Check < Hbestv, x1, Hbest)
                # Hbestv = np.where(Check < Hbestv, Check, Hbestv)
# 
            # elif i % 7 == 5:
                # x0 = x0.reshape(Size,)
                # Rns.shuffle(x0)
                # x0 = x0.reshape(MemNum, self.Digits)
                # with Pool(Cores) as p:
                    # Check = np.array(p.map(Obj_Func, x0)).reshape(MemNum, 1)
                    # p.close()
                    # p.join()
                # Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
                # Hbest=np.where(Check < Hbestv, x0, Hbest)
                # Hbestv = np.where(Check < Hbestv, Check, Hbestv)
# 
            # elif i % 7 == 1:
                # x1 = Hbest.copy()
                # x1 = x1.reshape(Size,)
                # Rns.shuffle(x1)
                # Index = Rns.randint(0, Size, int(Size / 3))
                # x1[Index] = 0
                # x1 = x1.reshape(MemNum, self.Digits)
                # with Pool(Cores) as p:
                    # Check = np.array(p.map(Obj_Func, x1)).reshape(MemNum, 1)
                    # p.close()
                    # p.join()
                # Check = np.array(list(map(Obj_Func, x1))).reshape(MemNum, 1)
                # Hbest=np.where(Check < Hbestv, x1, Hbest)
                # Hbestv = np.where(Check < Hbestv, Check, Hbestv)
            # else:
                # x0 = x0.reshape(Size,)
                # Rns.shuffle(x0)
                # Index = Rns.randint(0, Size, int(Size / 3))
                # x0[Index] = 0
                # x0 = x0.reshape(MemNum, self.Digits)
                # with Pool(Cores) as p:
                    # Check = np.array(p.map(Obj_Func, x1)).reshape(MemNum, 1)
                    # p.close()
                    # p.join()
                # Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
                # Hbest=np.where(Check < Hbestv, x0, Hbest)
                # Hbestv = np.where(Check < Hbestv, Check, Hbestv)


            Check = np.array(list(map(Obj_Func, x0))).reshape(MemNum, 1)
            Hbest=np.where(Check < Hbestv, x0, Hbest)
            Hbestv=np.where(Check < Hbestv, Check, Hbestv)
            print(i + 1)
            print("====")
            print(Hbestv.min())
            print(Hbest[np.argmin(Hbestv),:])
            print("")
            Data = pd.DataFrame(Hbest[np.argmin(Hbestv)])
            Data.to_excel("Data.xlsx",sheet_name="Data")
        return Hbestv.min(),Hbest[np.argmin(Hbestv),:]






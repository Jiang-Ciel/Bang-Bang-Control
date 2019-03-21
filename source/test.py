from multiprocessing import Pool
import multiprocessing
import time
import numpy as np
from Kinetic import Kinetic

inputs = np.random.rand(45, 21) * 100 - 50
scale = np.ones(45, dtype="int") * 250
cores = multiprocessing.cpu_count()
if __name__ == "__main__":
    test = Kinetic(21)
    t1 = time.time()
    with Pool(cores) as p:
        result = p.map(test.Obj_Func,inputs)
        p.close()
        p.join()
    t2 = time.time()
    result2 = np.array(list(map(test.Obj_Func, inputs)))
    result = np.array(result)
    t3 = time.time()
   
    print("With multiprocess", t2 - t1)
    print("Without mutiprocess", t3 - t2)
    
    
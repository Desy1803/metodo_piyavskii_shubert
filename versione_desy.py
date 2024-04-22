import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time

minimo=float('inf')

def selection_function(func_name):
    global f
    global t
    if func_name == 'f1':
        f = lambda x : (1/6)*x**6-(52/25)*x**5+(39/80)*x**4+(71/10)*x**3-(79/20)*x**2-x+(1/10)
        t = (-1.5, 11, 13870)
    elif func_name == 'f2':
        f = lambda x : math.sin(x)+math.sin(10*x/3)
        t = (2.7, 7.5, 4.29)
    elif func_name == 'f3':
        f = lambda x: sum(k*math.sin((k+1)*x+k) for k in range(1, 6))
        t = (-10, 10, 67)
    elif func_name == 'f4':
        f = lambda x : -(16*x**2-24*x+5)*math.exp(-x)
        t = (1.9, 3.9, 3)
    elif func_name == 'f5':
        f = lambda x : (3*x-1.4)*math.sin(18*x)
        t = (0, 1.2, 36)
    elif func_name == 'f6':
        f = lambda x : -(x+math.sin(x))*math.exp(-(x**2))
        t = (-10, 10, 2.5)
    elif func_name == 'f7':
        f = lambda x : math.sin(x)+math.sin(10*x/3)+math.log(x)-0.84*x+3
        t = (2.7, 7.5, 6)
    elif func_name == 'f8':
        f = lambda x: sum(k*math.cos((k+1)*x+k) for k in range(1, 6))
        t = (-10, 10, 67)
    elif func_name == 'f9':
        f = lambda x : math.sin(x)+math.sin(2*x/3)
        t = (3.1, 20.4, 1.7)
    elif func_name == 'f10':
        f = lambda x : -x*math.sin(x)
        t = (0, 10, 11)
    elif func_name == 'f11':
        f = lambda x : 2*math.cos(x)+math.cos(2*x)
        t = (-1.57, 6.28, 3)
    elif func_name == 'f12':
        f = lambda x : (math.sin(x))**3+(math.cos(x))**3
        t = (0, 6.28, 2.2)
    elif func_name == 'f13':
        f = lambda x : -(x**(2/3))+math.cbrt(x**2-1)
        t = (0.001, 0.99, 8.5)
    elif func_name == 'f14':
        f = lambda x : -math.exp(-x)*math.sin(2*math.pi*x)
        t = (0, 4, 6.5)
    elif func_name == 'f15':
        f = lambda x : (x**2-5*x+6)/(x**2+1)
        t = (-5, 5, 6.5)
    elif func_name == 'f16':
        f = lambda x : 2*(x-3)**2+math.exp(1)**(0.5*x**2)
        t = (-3, 3, 85)
    elif func_name == 'f17':
        f = lambda x : x**6-15*x**4+27*x**2+250
        t = (-4, 4, 2520)
    elif func_name == 'f18':
        f = lambda x : (x-2)**2 if x<=3 else 2*math.log(x-2)+1
        t = (0, 6, 4)
    elif func_name == 'f19':
        f = lambda x : -x+math.sin(3*x)-1
        t = (0, 6.5, 4)
    elif func_name == 'f20':
        f = lambda x : (math.sin(x)-x)*math.exp(-(x**2))
        t = (-10, 10, 1.3)

def selection_function_np(func_name):
    global fnp
    global tnp
    if func_name == 'f1':
        fnp = lambda x: (1/6)*x**6-(52/25)*x**5+(39/80)*x**4+(71/10)*x**3-(79/20)*x**2-x+(1/10)
        tnp = (-1.5, 11, 13870)
    elif func_name == 'f2':
        fnp = lambda x: np.sin(x)+np.sin(10*x/3)
        tnp = (2.7, 7.5, 4.29)
    elif func_name == 'f3':
        fnp = lambda x: sum(k*np.sin((k+1)*x+k) for k in range(1, 6))
        tnp = (-10, 10, 67)
    elif func_name == 'f4':
        fnp = lambda x: -(16*x**2-24*x+5)*np.exp(-x)
        tnp = (1.9, 3.9, 3)
    elif func_name == 'f5':
        fnp = lambda x: (3*x-1.4)*np.sin(18*x)
        tnp = (0, 1.2, 36)
    elif func_name == 'f6':
        fnp = lambda x: -(x+np.sin(x))*np.exp(-(x**2))
        tnp = (-10, 10, 2.5)
    elif func_name == 'f7':
        fnp = lambda x: np.sin(x)+np.sin(10*x/3)+np.log(x)-0.84*x+3
        tnp = (2.7, 7.5, 6)
    elif func_name == 'f8':
        fnp = lambda x: sum(k*np.cos((k+1)*x+k) for k in range(1, 6))
        tnp = (-10, 10, 67)
    elif func_name == 'f9':
        fnp = lambda x: np.sin(x)+np.sin(2*x/3)
        tnp = (3.1, 20.4, 1.7)
    elif func_name == 'f10':
        fnp = lambda x: -x*np.sin(x)
        tnp = (0, 10, 11)
    elif func_name == 'f11':
        fnp = lambda x: 2*np.cos(x)+np.cos(2*x)
        tnp = (-1.57, 6.28, 3)
    elif func_name == 'f12':
        fnp = lambda x: (np.sin(x))**3+(np.cos(x))**3
        tnp = (0, 6.28, 2.2)
    elif func_name == 'f13':
        fnp = lambda x: -(x**(2/3))+np.cbrt(x**2-1)
        tnp = (0.001, 0.99, 8.5)
    elif func_name == 'f14':
        fnp = lambda x: -np.exp(-x)*np.sin(2*np.pi*x)
        tnp = (0, 4, 6.5)
    elif func_name == 'f15':
        fnp = lambda x: (x**2-5*x+6)/(x**2+1)
        tnp = (-5, 5, 6.5)
    elif func_name == 'f16':
        fnp = lambda x: 2*(x-3)**2+np.exp(1)**(0.5*x**2)
        tnp = (-3, 3, 85)
    elif func_name == 'f17':
        fnp = lambda x: x**6-15*x**4+27*x**2+250
        tnp = (-4, 4, 2520)
    elif func_name == 'f18':
        fnp = lambda x: np.where(x <= 3, (x-2)**2, np.where(x > 2, 2*np.log(x-2)+1, np.nan))
        tnp = (0, 6, 4)
    elif func_name == 'f19':
        fnp = lambda x: -x+np.sin(3*x)-1
        tnp = (0, 6.5, 4)
    elif func_name == 'f20':
        fnp = lambda x: (np.sin(x)-x)*np.exp(-(x**2))
        tnp = (-10, 10, 1.3)

#############################ALGORITMO###########################
def fun(x):
    return fnp(x)

def z(x):
    global minimo
    val = f(x)
    if val < minimo:
        minimo = val
    return val
#                 CALCOLO R
def calcola_R(x_interval, L):
    R = z(x_interval[0]) + z(x_interval[1]) - L * (x_interval[1] - x_interval[0]) / 2
    return R
#                 TROVO MINR
def trova_min_Ri(x_intervals, L):
    minimo1 = float('inf')
    minimo2 = float('inf')
    min_idx1 = None
    min_idx2 = None
    
    for i in range(len(x_intervals)):
        Ri = calcola_R(x_intervals[i], L)
        
        if Ri < minimo1:
            minimo2 = minimo1
            min_idx2 = min_idx1
            minimo1 = Ri
            min_idx1 = i
        elif Ri < minimo2 and min_idx1 != i:
            minimo2 = Ri
            min_idx2 = i
    
    min_values = [(min_idx1, minimo1)]
    if min_idx2 is not None:
        min_values.append((min_idx2, minimo2))
    
    return min_values

def flatten(intervals):
    if len(intervals) == 0:
        return None
    values = [intervals[0][0]]
    for interval in intervals:
        values.append(interval[1])
    return values

def metodo_piyavskii_shubert(a, b, L, epsilon):
    global minimo
    k = 1
    maxIter = 12
    x_intervals = [(a, b)]
    
    while True:
        #print("Iterazione \n ", k)
        interval2 = None
        min_values = trova_min_Ri(x_intervals, L)
        #print("min_values: ", min_values)
        interval1 = x_intervals[min_values[0][0]] 
        xt_new1 = ((interval1[0] + interval1[1]) / 2) - ((z(interval1[1]) - z(interval1[0])) / (2 * L))
        
        if len(min_values) == 1 and min_values[0][1] > minimo:
            print("qui")
            break
        x_intervals.remove(interval1)
        if min_values[0][1] < minimo:
            t = min_values[0][0]
            x_intervals.insert(t, (interval1[0], xt_new1))
            x_intervals.insert(t+1, (xt_new1, interval1[1]))
        #print("x_intervals between: ", x_intervals)
        if len(min_values) > 1 and len(x_intervals) > 1:
            interval2 = x_intervals[min_values[1][0]]
            xt_new2 = ((interval2[0] + interval2[1]) / 2) - ((z(interval2[1]) - z(interval2[0])) / (2 * L))
            if len(min_values) == 1 and min_values[1][0] > minimo:
                print("qui2")
                break
            x_intervals.remove(interval2)
            if min_values[1][1] < minimo:
                t = min_values[1][0]
                x_intervals.insert(t, (interval2[0], xt_new2))
                x_intervals.insert(t, (xt_new2, interval2[1]))
        #print("x_intervals after: ", x_intervals)
        if interval1[1] - interval1[0] <= epsilon:
            if interval2 != None and interval2[1] - interval2[0] <= epsilon :
                x_values = flatten(x_intervals)
                print("Numero di iterazioni con int2:", k)
                return x_values, [z(x) for x in x_values]
                
            if interval2 == None :
                x_values = flatten(x_intervals)
                print("Numero di iterazioni:", k)
                return x_values, [z(x) for x in x_values]

        k += 1
    print("Numero di iterazioni fuori:", k)    
    x_values = flatten(x_intervals)
    
    return x_values, [z(x) for x in x_values]

def main(func_name):
    selection_function(func_name)
    selection_function_np(func_name)
    global t
    global minimo
    a = t[0]
    b = t[1]
    L = t[2]
    epsilon = 0.00001
    
    start_time = time.time()
    x_values, z_values = metodo_piyavskii_shubert(a, b, L, epsilon)
    end_time = time.time()
    execution_time = end_time - start_time

    print("Tempo di esecuzione:", execution_time, "secondi")
    minimo_z = min(z_values)
    min_idx = z_values.index(minimo_z)
    minimo_x = x_values[min_idx]
    print("x: ", minimo_x)
    print("y: ", minimo_z)
    intervallo_valori_plot = np.linspace(a, b, 10000)
    y = fun(intervallo_valori_plot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(intervallo_valori_plot, y, color='blue', linestyle='-', label='Funzione')
    ax.scatter(x_values, z_values, color='red', label='')
    ax.scatter(minimo_x, minimo_z, color='green', label='Minimo globale')
    ax.set_title('Grafico')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.grid(True)
    plt.show()
    
main("f19")

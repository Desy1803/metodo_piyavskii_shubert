import math
import numpy as np
import matplotlib.pyplot as plt
import time

fcost = lambda x: 1
tcost = (-1,1,2.5)

f1 = lambda x : (1/6)*x**6-(52/25)*x**5+(39/80)*x**4+(71/10)*x**3-(79/20)*x**2-x+(1/10)
t1 = (-1.5, 11, 13870)
f2 = lambda x : math.sin(x)+math.sin(10*x/3)
t2 = (2.7, 7.5, 4.29)
f3 = lambda x: sum(k*math.sin((k+1)*x+k) for k in range(1, 6))
t3 = (-10, 10, 67)
f4 = lambda x : -(16*x**2-24*x+5)*math.exp(-x)
t4 = (1.9, 3.9, 3)
f5 = lambda x : (3*x-1.4)*math.sin(18*x)
t5 = (0, 1.2, 36)
f6 = lambda x : -(x+math.sin(x))*math.exp(-(x**2))
t6 = (-10, 10, 2.5)
f7 = lambda x : math.sin(x)+math.sin(10*x/3)+math.log(x)-0.84*x+3
t7 = (2.7, 7.5, 6)
f8 = lambda x: sum(k*math.cos((k+1)*x+k) for k in range(1, 6))
t8 = (-10, 10, 67)
f9 = lambda x : math.sin(x)+math.sin(2*x/3)
t9 = (3.1, 20.4, 1.7)
f10 = lambda x : -x*math.sin(x)
t10 = (0, 10, 11)
f11 = lambda x : 2*math.cos(x)+math.cos(2*x)
t11 = (-1.57, 6.28, 3)
f12 = lambda x : (math.sin(x))**3+(math.cos(x))**3
t12 = (0, 6.28, 2.2)
f13 = lambda x : -(x**(2/3))+math.cbrt(x**2-1)
t13 = (0.001, 0.99, 8.5)
f14 = lambda x : -math.exp(-x)*math.sin(2*math.pi*x)
t14 = (0, 4, 6.5)
f15 = lambda x : (x**2-5*x+6)/(x**2+1)
t15 = (-5, 5, 6.5)
f16 = lambda x : 2*(x-3)**2+math.exp(1)**(0.5*x**2)
t16 = (-3, 3, 85)
f17 = lambda x : x**6-15*x**4+27*x**2+250
t17 = (-4, 4, 2520)
f18 = lambda x : (x-2)**2 if x<=3 else 2*math.log(x-2)+1
t18 = (0, 6, 4)
f19 = lambda x : -x+math.sin(3*x)-1
t19 = (0, 6.5, 4)
f20 = lambda x : (math.sin(x)-x)*math.exp(-(x**2))
t20 = (-10, 10, 1.3)




#---------------------------------------ALGORITMO--------------------------------------
minimo = float('inf')
a = 0
b = 0
L = 0
def fun(x):
      y=[]
      for i in range(len(x)):
           y.append(fcost(x[i]))                          #RICORDA
      return y


def z(x):
    global minimo
    val = fcost(x)                                        #RICORDA!
    if(val < minimo):
        minimo = val
    return val


def calcola_R(x_interval, L):
    R = z(x_interval[0]) + z(x_interval[1]) - L * (x_interval[1] - x_interval[0]) / 2
    #print(f"Calcolato R per intervallo {x_interval}: {round(R, 2)}")
    return R


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
    # appiattisce array di intervalli in array di valori
    if len(intervals)==0:
        return None
    values = [intervals[0][0]]  # lower del primo intervallo
    for interval in intervals:
        values.append(interval[1])  # tutti gli upper
    return values


def metodo_piyavskii_shubert(a, b, L, epsilon):
    global minimo
    k = 1
    maxIter = 12
    x_intervals = [(a, b)]  # array di intervalli
    
    while True:
        #print("Numero di iterazione:" , k)
        interval2 = None
        min_values = trova_min_Ri(x_intervals, L)
        interval1 = x_intervals[min_values[0][0]] 
        xt_new1 = ((interval1[0] + interval1[1]) / 2) - ((z(interval1[1]) - z(interval1[0])) / (2 * L))
        #print("xt1_new", xt_new1)
        #print("x_intervals before:", x_intervals)
        #print("min_values:", min_values)
        # dal punto xt_new, crea due nuovi intervalli e rimuovi il precedente
        if len(min_values)==1 and min_values[0][1]> minimo:
            break
        x_intervals.remove(interval1)
        if min_values[0][1] < minimo:
            t = min_values[0][0]
            x_intervals.insert(t, (interval1[0], xt_new1))
            x_intervals.insert(t+1, (xt_new1, interval1[1]))

        if len(min_values)>1 and len(x_intervals)>1:
            #print("between",x_intervals)
            interval2 = x_intervals[min_values[1][0]]
            xt_new2 = ((interval2[0] + interval2[1]) / 2) - ((z(interval2[1]) - z(interval2[0])) / (2 * L))
            #print("xt2_new", xt_new2)
            if len(min_values)==1 and min_values[1][0]> minimo:
                break
            x_intervals.remove(interval2)
            if min_values[1][1]< minimo:
                t = min_values[1][0]
                x_intervals.insert(t, (interval2[0], xt_new2))
                x_intervals.insert(t, (xt_new2, interval2[1]))
        #print("x_intervals after:", x_intervals)
        if interval1[1] - interval1[0] <= epsilon :
            if interval2 != None and interval2[1] - interval2[0] <= epsilon :
                x_values = flatten(x_intervals)
                print("numero di iterazioni", k)
                return x_values, [z(x) for x in x_values]
            if interval2 == None:
                x_values = flatten(x_intervals)
                print("numero di iterazioni", k)
                return x_values, [z(x) for x in x_values]

        k += 1
        
    x_values = flatten(x_intervals)
    return x_values, [z(x) for x in x_values]

def main():
    global minimo
    a = tcost[0]
    b = tcost[1]
    L = tcost[2]
    epsilon = 0.00001
    print("precisione", epsilon)
    
    # Esecuzione dell'algoritmo e plot dei punti trovati
    start_time = time.time()
    x_values, z_values = metodo_piyavskii_shubert(a, b, L, epsilon)  # Assign return values here
    end_time = time.time()
    execution_time = end_time - start_time

    print("Tempo di esecuzione:", execution_time, "secondi")
    minimo_z = min(z_values)
    min_idx = z_values.index(minimo_z)
    minimo_x = x_values[min_idx]
    print("x: ", minimo_x)
    print("y: ", minimo_z)
    intervallo_valori_plot = np.linspace(a, b, 100000)
    y = fun(intervallo_valori_plot)
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot della funzione obiettivo
    ax.plot(intervallo_valori_plot, y, color='blue', linestyle='-', label='Funzione')
    
    
    # Plot dei punti trovati durante l'esecuzione dell'algoritmo
    ax.scatter(x_values, z_values, color='red', label='')
    
    # Plot del minimo globale
    ax.scatter(minimo_x, minimo_z, color='green', label='Minimo globale')
    
    # Impostazione del titolo e delle etichette degli assi
    ax.set_title('Grafico ')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Mostra la legenda
    ax.legend()
    
    # Mostra il grafico
    plt.grid(True)
    plt.show()

 

main()

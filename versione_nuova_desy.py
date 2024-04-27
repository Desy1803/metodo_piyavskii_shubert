import math
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq

#SELEZIONE FUNZIONE

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
#                       ALGORITMO

class Punto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __lt__(self, other):
        if self.y == other.y:
            return self.x < other.x
        else:
            return self.y < other.y


#               heap ordinamento in base ad R

class IntervalHeap:
    def __init__(self):
        self.heap = []
        self.size = 0

    def insert(self, R, interval):
        heapq.heappush(self.heap, (R, interval))
        self.size += 1

    def pop(self):
        return heapq.heappop(self.heap)[1]  # Restituisce solo l'intervallo, ignorando il valore di R e l'identificatore
    
    def is_empty(self):
        return len(self.heap) == 0
    
    def __str__(self):
        return ', '.join([f"(R: {R},I: {interval})" for R, interval in self.heap])
    
    def __lt__(self, nxt):
        if self.heap[0][0] == nxt.heap[0][0]:
            return self.heap[0][1][0] < nxt.heap[0][1][0]
        else:
            return self.heap[0][0] < nxt.heap[0][0]


potatura=0

#                       PLOT FUNZIONE
def fun(x):
    return fnp(x)

#                       CALCOLO R
def calcola_R(x_interval, L):
    R = f(x_interval[0]) + f(x_interval[1]) - L * (x_interval[1] - x_interval[0]) / 2
    return R

def metodo_piyavskii_shubert(a,b,L,epsilon):
    global minimo_y
    global minimo_x
    global k
    global potatura
    values = []
    heap = IntervalHeap()
    k = 1
    R = calcola_R((a,b), L)
    heap.insert(R, (a, b))
    minimo_y=float('inf')
    while True:
        #print("Heap ", heap)
       # print(" ")
        # Estrai l'intervallo minimo (quello con il minimo di R)
        if heap.is_empty():
            return values
        interval1 = heap.pop()
        #criterio di terminazione
        if interval1[1] - interval1[0] <= epsilon or interval1 is None:
            return values
        
        xt_new1 = ((interval1[0] + interval1[1]) / 2) - ((f(interval1[1]) - f(interval1[0])) / (2 * L))
        if minimo_y is None or f(xt_new1) < minimo_y:  # Se minimo è None o xt_new1 è più piccolo del minimo attuale
            minimo_y =f(xt_new1)  # Aggiorna minimo
            minimo_x = xt_new1 
        # Inserisci il nuovo intervallo a sinistra dell'intervallo estratto
        R1 = calcola_R((interval1[0], xt_new1), L)
        if R1<minimo_y:
            heap.insert(R1, (interval1[0], xt_new1))
        else:
            potatura+=1
        # a dx
        R2 = calcola_R((xt_new1, interval1[1]), L)
        if R2<minimo_y:
            heap.insert(R2,(xt_new1, interval1[1]))
        else:
            potatura+=1
        #valori calcolati
        values.append(Punto(xt_new1, f(xt_new1)))
        
        

        k+=1
    
    return values

#                       MAIN
def main(func_name):
    global k
    global t
    global minimo_y
    global minimo_x
    global potatura
        
    selection_function(func_name)
    selection_function_np(func_name)
    
    a = t[0]
    b = t[1]
    L = t[2]
    epsilon = 0.00001
    
    start_time = time.time()
    values = metodo_piyavskii_shubert(a, b, L, epsilon)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Funzione: ", func_name)
    print("Epsilon: ", epsilon)
    print("Numero di iterazioni: ", k)
    print("Tempo di esecuzione:", execution_time, "secondi")
    print("Potatura: ", potatura)
    print("x: ", minimo_x)
    print("y: ", minimo_y)
    x_values = [punto.x for punto in values]
    z_values = [punto.y for punto in values]
    
    intervallo_valori_plot = np.linspace(a, b, 1000000)
    y = fun(intervallo_valori_plot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(intervallo_valori_plot, y, color='blue', linestyle='-', label='Funzione')
    ax.scatter(x_values, z_values, color='red', label='')
    ax.scatter(minimo_x, minimo_y, color='green', label='Minimo globale')
    ax.set_title('Grafico')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.grid(True)
    plt.show()

   
main("f20")


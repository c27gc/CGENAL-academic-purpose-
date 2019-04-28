from cgenalcore import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib import collections  as mc
import pylab as pl
from time import sleep
import numpy as np


f = 10
c = 10
numero_de_elementos = f*c
tamano_del_cromosoma = 12
elitism = "direct-elitism"
probabilidad_de_cruce = 0.9
probabilidad_de_mutacion = 0.002
mutacion = "evaluated-npermutation"
problema = 7
gh = []
fitAverages = []
thebest = []
mutation = []
iteraciones = 5000
punto_de_cruce = random.randint(1,11)

posiciones = {1:(0,0),2:(0,1),3:(0,2),4:(0,3),
5:(1,3),6:(2,3),7:(3,3),8:(3,2),
9:(3,1),10:(3,0),11:(2,0),12:(1,0)}



a = InitialPopulation(numero_de_elementos, tamano_del_cromosoma, problema)
fitAverages.append(a.get_fitness_average())
thebest.append(a.get_best_individual().get_fitness())

#AQUI TRANSFORMA LAS LISTAS EN ARRAY Y LAS EXTIENDE

aut = a.get_chromo_map()
fit = a.get_fit_map()
autEx = a.get_e_chromo_map()
fitEx = a.get_e_fit_map()


#AQUI SE DEFINE LA POLITICA DE SELECCIOND DE PADRES
dx = {0:-1, 1:0, 2:1, 3:0}
dy = {0:0, 1:-1, 2:0, 3:1}

p2 = np.zeros((f,c,tamano_del_cromosoma))
p2fit = np.zeros((f,c))

for i in range(0,f):
    for j in range(0,c):
        t = np.random.randint(0,4)
        #print("t: {}\ndx[t]: {}\ndy[t]: {}".format(t,dx[t],dy[t]))
        x = i + 1 + dx[t]
        y = j + 1 + dy[t]
        p2[i,j] = autEx[x,y]
        p2fit[i,j] = fitEx[x,y]



#CRUCE Y MUTACION
b = Crossing(a,punto_de_cruce,probabilidad_de_cruce,probabilidad_de_mutacion,elitism,mutacion,problema)
C,mt = b.sCross(aut,p2)
C = np.array(C)
K = C.reshape(-1,C.shape[-1])

mutation.append(b.get_mutation())
count=1
gh.append(count)

fig, (ax1, ax2) = plt.subplots(1,2)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
print('Trabajando...')

while(count!=iteraciones):

    childs = Generation(K.tolist(),8)
    fitAverages.append(childs.get_fitness_average())
    thebest.append(childs.get_best_individual().get_fitness())

    if count % int(iteraciones/100) == 1:
        aut = np.ones((f,c,tamano_del_cromosoma))
        aut = aut.tolist()
        fit = np.ones((f,c))
        fit = fit.tolist()
        k = 0
        for i in range(0,f):
            for j in range(0,c):
                aut[i][j]=np.asarray(childs.get_individuals()[k].chromo)
                fit[i][j]=np.asarray(childs.get_individuals()[k].get_fitness())
                k+=1
        fit = np.array(fit)
        aut = np.array(aut)
        """fig, ax = plt.subplots(211)
        min_val, max_val = 0, 1
        print(type(ax))"""

        ax2.matshow(fit, cmap=plt.cm.Blues)

        """for i in xrange(4):
            for j in xrange(5):
                c = fit[j,i]
                ax.text(i, j, str(c), va='center', ha='center')"""
        """plt.show()
        plt.close(1)"""


        print("Iteracion = {}\n Mejor Individuo = {}".format(count,childs.get_best_individual().chromo))
        lines=[]
        for i in range(0,len(childs.get_best_individual().chromo)-1):
            h=[]
            g=[]
            h.append(posiciones[childs.get_best_individual().chromo[i]])
            h.append(posiciones[childs.get_best_individual().chromo[i+1]])
            lines.append(h)
            h=[]
        #lines.append((lines[0][0],lines[11][1]))
        lc = mc.LineCollection(lines, linewidths=2)

        ax1.add_collection(lc)
        ax1.autoscale()
        ax1.margins(0.1)

        ax1.scatter(0,0, s=10)
        ax1.text(0,0, s=1)
        ax1.scatter(0,1, s=10)
        ax1.text(0,1, s=2)
        ax1.scatter(0,2, s=10)
        ax1.text(0,2, s=3)
        ax1.scatter(0,3, s=10)
        ax1.text(0,3, s=4)
        ax1.scatter(1,3, s=10)
        ax1.text(1,3, s=5)
        ax1.scatter(2,3, s=10)
        ax1.text(2,3, s=6)
        ax1.scatter(3,3, s=10)
        ax1.text(3,3, s=7)
        ax1.scatter(3,2, s=10)
        ax1.text(3,2, s=8)
        ax1.scatter(3,1, s=10)
        ax1.text(3,1, s=9)
        ax1.scatter(3,0, s=10)
        ax1.text(3,0, s=10)
        ax1.scatter(2,0, s=10)
        ax1.text(2,0, s=11)
        ax1.scatter(1,0, s=10)
        ax1.text(1,0, s=12)
        """plt.show()
        plt.close(1)"""



        plt.show(block=False)
        plt.pause(1)
        ax1.clear()
        ax2.clear()
    #for i in range(0,12):


    punto_de_cruce = random.randint(1,11)
    #punto_de_cruce = 6
    Cross = Crossing(childs,punto_de_cruce,probabilidad_de_cruce,probabilidad_de_mutacion,elitism,mutacion,problema)
    C,mt = Cross.sCross(aut,p2)
    C = np.array(C)
    K = C.reshape(-1,C.shape[-1])
    #mutation.append(Cross.get_mutation())
    count+=1
    gh.append(count)
    #print(count)

print('Hecho: \n'+ 'Mutaciones: ' + str(sum(mutation))+ "\n" + 'Mejor Solución: ' + str(childs.get_best_individual().chromo) + '\nMejor Solución(Distancia): ' + str(childs.get_best_individual().get_fitness()**(-1)*12)+'\nFitness de la Mejor Solución: ' + str(childs.get_best_individual().get_fitness())+'\nIteraciones: ' + str(count) )
plt.close()

x=np.asarray(fitAverages)
x2=np.asarray(thebest)
x3=np.asarray(mutation)
gh=np.asarray(gh)


ax1 = plt.subplot(311)
ax1.set_ylim(np.amin(x)*0.9,np.amax(x)*1.1)
ax1.grid(True)
ax1.set_ylabel("Fitness\n Promedio")
ax1.plot(gh,x)

ax2 = plt.subplot(312)
ax2.set_ylim(np.amin(x2)*0.9,np.amax(x2)*1.1)
ax2.grid(True)
ax2.set_ylabel("Fitness\n Mejor Individuo")
ax2.plot(gh,x2)

ax3 = plt.subplot(313)
ax3.set_ylim(0,int(np.amax(x3)+1))
ax3.grid(True)
ax3.set_ylabel("Número de\n Mutaciones")
ax3.set_xlabel("Iteraciones")
ax3.plot(gh,x3)
#plt.show()

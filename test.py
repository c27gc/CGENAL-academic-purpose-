from cgenalcore import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib import collections  as mc
import pylab as pl
from time import sleep


def extendedMatrix(m):
    s=np.shape(m)
    B=[]
    A=m
    if len(s) > 3 or len(s) < 2:
        print("Error, la matriz debe ser al menos de 2 dimensiones y maximo de 3 demensioens.")
    else:
        if len(s) == 2:
            B=np.pad(A,((1, 1), (1, 1)), 'wrap')
        if len(s) == 3:
            B=np.pad(A,((1, 1), (1, 1), (0,0)), 'wrap')

        B[0,0]=0
        B[0,s[1]+1]=0
        B[s[0]+1,0]=0
        B[s[0]+1,s[1]+1]=0
        print(type(B))
    return B

numero_de_elementos = 20
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
aut = np.ones((4,5,12))
bk = np.zeros((4,5,12))
fit = np.ones((4,5))
k = 0

for i in range(0,4):
    for j in range(0,5):
        aut[i][j]=np.asarray(a.get_individuals()[k].chromo)
        fit[i][j]=np.asarray(a.get_individuals()[k].get_fitness())
        k+=1

#AQUI SE DEFINE LA POLITICA DE SELECCIOND DE PADRES
autEx = extendedMatrix(aut)
fitEx = extendedMatrix(fit)

print(np.shape(autEx))

dx = {0:-1, 1:0, 2:1, 3:0}
dy = {0:0, 1:-1, 2:0, 3:1}

p2 = np.zeros((4,5,12))
p2fit = np.zeros((4,5))

for i in range(0,4):
    for j in range(0,5):
        t=np.random.randint(0,4)
        print("t: {}\ndx[t]: {}\ndy[t]: {}".format(t,dx[t],dy[t]))
        x = i + 1 + dx[t]
        y = j + 1 + dy[t]
        p2[i,j] = autEx[x,y]
        p2fit[i,j] = fitEx[x,y]
#print(fitEx)
fitAverages.append(a.get_fitness_average())
thebest.append(a.get_best_individual().get_fitness())
b = Crossing(a,punto_de_cruce,probabilidad_de_cruce,probabilidad_de_mutacion,elitism,mutacion,problema)
b.cross()
print("_______________________________________")
parents2 = b.get_childs()
for i in range(0,len(parents2)):
    print(parents2[i])


mutation.append(b.get_mutation())
count=1
gh.append(count)
print('Trabajando...')

while(count!=iteraciones):
    childs = Generation(parents2,problema)
    fitAverages.append(childs.get_fitness_average())
    thebest.append(childs.get_best_individual().get_fitness())
    if count % int(iteraciones/100) == 1:
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
        fig, ax = pl.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        plt.scatter(0,0, s=10)
        plt.text(0,0, s=1)
        plt.scatter(0,1, s=10)
        plt.text(0,1, s=2)
        plt.scatter(0,2, s=10)
        plt.text(0,2, s=3)
        plt.scatter(0,3, s=10)
        plt.text(0,3, s=4)
        plt.scatter(1,3, s=10)
        plt.text(1,3, s=5)
        plt.scatter(2,3, s=10)
        plt.text(2,3, s=6)
        plt.scatter(3,3, s=10)
        plt.text(3,3, s=7)
        plt.scatter(3,2, s=10)
        plt.text(3,2, s=8)
        plt.scatter(3,1, s=10)
        plt.text(3,1, s=9)
        plt.scatter(3,0, s=10)
        plt.text(3,0, s=10)
        plt.scatter(2,0, s=10)
        plt.text(2,0, s=11)
        plt.scatter(1,0, s=10)
        plt.text(1,0, s=12)
        plt.show()
        plt.close(1)
    #for i in range(0,12):


    punto_de_cruce = random.randint(1,11)
    #punto_de_cruce = 6
    Cross = Crossing(childs,punto_de_cruce,probabilidad_de_cruce,probabilidad_de_mutacion,elitism,mutacion,problema)
    Cross.cross()
    parents2 = Cross.get_childs()
    mutation.append(Cross.get_mutation())
    count+=1
    gh.append(count)
    #print(count)

print('Hecho: \n'+ 'Mutaciones: ' + str(sum(mutation))+ "\n" + 'Mejor Solución: ' + str(childs.get_best_individual().chromo) + '\nMejor Solución(Distancia): ' + str(childs.get_best_individual().get_fitness()**(-1)*12)+'\nFitness de la Mejor Solución: ' + str(childs.get_best_individual().get_fitness())+'\nIteraciones: ' + str(count) )

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Modulo Creado por Carlos E. Gonzalez C.
#Para la materia Algoritmos Geneticos punto
import random
from fitnessdef import *
import copy
import time
import math
import numpy as np
import matplotlib.pyplot as plt

class InitialPopulation:
    """docstring for Population."""
    __individuals = []
    __fitness = []
    __fitness_average = 0
    __best_individual = []
    __second_best_individual = []

    def __init__(self, number_of_individuals, size_of_chromosomes, problem, f=10, c=10):
        self.number_of_individuals = number_of_individuals
        self.size_of_chromosomes = size_of_chromosomes
        #Creating individual and list of fitness
        self.__individuals = []
        self.__fitness = []
        if (problem == 1) or (problem == 2):
            for i in range(0,number_of_individuals):
                chromosome=[]
                for j in range(0,size_of_chromosomes):
                    chromosome.append(random.randint(0,1))
                self.__individuals.append(ChromosomeClass(chromosome, problem))
                self.__fitness.append(self.__individuals[i].get_fitness())

        if (problem == 3):
            print("{}\t {}".format("Cromosoma", "Fitness"))
            for i in range(0,number_of_individuals):
                indv = []
                k=random.uniform(0,1)
                w=math.asin(k)
                indv = self.z2b(self.r2z(w,3.14159265359,0,16))
                self.__individuals.append(ChromosomeClass(indv, problem))
                self.__fitness.append(self.__individuals[i].get_fitness())

                print("{}\t {}".format(self.__individuals[i].chromo, self.truncate(self.__individuals[i].get_fitness(),4)))

        #"""CODIGO NUEVOOOOOO"""
        if (problem == 7 or problem == 8):
            a = list(range(1,size_of_chromosomes + 1))
            for i in range(0,number_of_individuals):
                chromosome=[]
                chromosome=random.sample(a,len(a))
                self.__individuals.append(ChromosomeClass(chromosome, problem))
                self.__fitness.append(self.__individuals[i].get_fitness())
                print("{}\t {}\t {}".format(chromosome,self.__individuals[i].get_fitness(),(self.__individuals[i].get_fitness()**(-1))*(120)))

            aut = np.ones((f,c,size_of_chromosomes))
            fit = np.ones((f,c))
            k = 0
            for i in range(0,f):
                for j in range(0,c):
                    aut[i][j] = np.asarray(self.__individuals[k].chromo)
                    fit[i][j] = np.asarray(self.__individuals[k].get_fitness())
                    k += 1
            self.__chromo_map = aut
            self.__fit_map = fit
            self.__e_chromo_map = extendedMatrix(aut)
            self.__e_fit_map = extendedMatrix(fit)

        #"""FIN CODIGO NUEVOOOOOO"""

        #Fitness Average Calculation
        self.__fitness_average = sum(self.__fitness)/(number_of_individuals)

        #Finding the best and second best individuals and his index
        fitness_copy=copy.deepcopy(self.__fitness)
        best_individual_index = fitness_copy.index(max(fitness_copy))
        self.__best_individual = self.__individuals[best_individual_index]
        fitness_copy[best_individual_index]=min(fitness_copy)
        best_individual_index = fitness_copy.index(max(fitness_copy))
        self.__second_best_individual = self.__individuals[best_individual_index]

    def get_chromo_map(self):
        return self.__chromo_map

    def get_e_chromo_map(self):
        return self.__e_chromo_map

    def get_fit_map(self):
        return self.__fit_map

    def get_e_fit_map(self):
        return self.__e_fit_map

    def get_individuals(self):
        return self.__individuals

    def get_fitness(self):
        return self.__fitness

    def get_fitness_average(self):
        return self.__fitness_average

    def get_best_individual(self):
        return self.__best_individual

    def get_second_best_individual(self):
        return self.__second_best_individual

    def __del__(self):
        self.__individuals = []
        self.__fitness = []
        self.__fitness_average = 0
        self.__best_individual = []
        self.__second_best_individual = []

    def r2z(self,r,rmax,rmin,l):
        z=((r-rmin)/(rmax-rmin))*((2**(l))-1)
        return int(z)

    def z2b(self,z):
        a=bin(int(z))
        a=a[2:]
        b=list(a)
        for m in range(len(b)):
            b[m]=int(b[m])
        while(1):
            if len(b)!=16:
                b.insert(0,0)
            else:
                break
        return b

    def truncate(self,number, digits) -> float:
        stepper = pow(10.0, digits)
        return math.trunc(stepper * number) / stepper

class Generation:
    """docstring for Generation."""
    __individuals = []
    __fitness = []
    __fitness_average = 0
    __best_individual = []
    __second_best_individual = []
    __worst_individual = []
    __second_worst_individual = []

    def __init__(self, list_of_individuals, problem, f=10,c=10):
        self.list_of_individuals = list_of_individuals
        number_of_individuals = len(list_of_individuals)
        size_of_chromosomes = len(list_of_individuals[0])

        #Creating individual and list of fitness
        self.__individuals = []
        self.__fitness = []
        for i in range(0,number_of_individuals):
            self.__individuals.append(ChromosomeClass(list_of_individuals[i],problem))
            self.__fitness.append(self.__individuals[i].get_fitness())

        #Fitness Average Calculation
        self.__fitness_average = sum(self.__fitness)/(number_of_individuals)

        #Finding the best and second best individuals and their indices
        fitness_copy=copy.deepcopy(self.__fitness)
        best_individual_index = fitness_copy.index(max(fitness_copy))
        self.__best_individual = self.__individuals[best_individual_index]
        fitness_copy[best_individual_index]=min(fitness_copy)
        best_individual_index = fitness_copy.index(max(fitness_copy))
        self.__second_best_individual = self.__individuals[best_individual_index]

        #finding the wost and second wost individuals and their indices
        fitness_copy=copy.deepcopy(self.__fitness)
        worst_individual_index = fitness_copy.index(max(fitness_copy))
        self.__worst_individual = self.__individuals[worst_individual_index]
        fitness_copy[worst_individual_index]=max(fitness_copy)
        worst_individual_index = fitness_copy.index(min(fitness_copy))
        self.__second_worst_individual = self.__individuals[worst_individual_index]

        if problem == 8:
            aut = np.ones((f,c,size_of_chromosomes))
            fit = np.ones((f,c))
            k = 0
            for i in range(0,f):
                for j in range(0,c):
                    aut[i][j] = np.asarray(self.__individuals[k].chromo)
                    fit[i][j] = np.asarray(self.__individuals[k].get_fitness())
                    k += 1
            self.__chromo_map = aut
            self.__fit_map = fit
            self.__e_chromo_map = extendedMatrix(aut)
            self.__e_fit_map = extendedMatrix(fit)

    def get_chromo_map(self):
        return self.__chromo_map

    def get_e_chromo_map(self):
        return self.__e_chromo_map

    def get_fit_map(self):
        return self.__fit_map

    def get_e_fit_map(self):
        return self.__e_fit_map

    def get_individuals(self):
        return self.__individuals

    def get_fitness(self):
        return self.__fitness

    def get_fitness_average(self):
        return self.__fitness_average

    def get_best_individual(self):
        return self.__best_individual

    def get_second_best_individual(self):
        return self.__second_best_individual

    def get_worst_individual(self):
        return self.__worst_individual

    def get_second_worst_individual(self):
        return self.__second_worst_individual

    def __del__(self):
        self.__individuals = []
        self.__fitness = []
        self.__fitness_average = 0
        self.__best_individual = []
        self.__second_best_individual = []
        self.__worst_individual = []
        self.__second_worst_individual = []

class Crossing():
    """docstring for Crossing."""
    __probability_value = 0
    __parents = []
    __childs = []
    __mutations = 0
    def __init__(self, list_of_individuals, cross_point, cross_probability, mutation_probability,elitism="non-elitism",mutation_type="normal",problem=8):
        AQUI1 = time.time()
        self.list_of_individuals = list_of_individuals
        self.cross_point = cross_point
        self.cross_probability = cross_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.mutation_type = mutation_type
        self.problem = problem

        #setprobability
        fitness = self.list_of_individuals.get_fitness()
        total_fitness = float(sum(fitness))
        relative_fitness = [f/total_fitness for f in fitness]
        self.__probability_value = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
        self.__mutations = 0

        if problem == 8:
            self.aut = self.list_of_individuals.get_chromo_map()
            self.fit = self.list_of_individuals.get_fit_map()
            self.autEx = self.list_of_individuals.get_e_chromo_map()
            self.fitEx = self.list_of_individuals.get_e_fit_map()
            self.f = np.shape(self.aut)[0]
            self.c = np.shape(self.aut)[1]
            self.tamano_del_cromosoma = np.shape(self.aut)[2]


    def __del__(self):
        self.list_of_individuals = 0
        self.cross_point = 0
        self.cross_probability = 0
        self.mutation_probability = 0
        self.elitism = 0
        self.mutation_type = 0
        self.problem = 0
        self.__probability_value = 0
        self.__parents = []
        self.__childs = []
        self.__mutations = 0

    def parent2(self, type = 'random', neighborhood = 'vonN'):

        #AQUI SE DEFINE LA POLITICA DE SELECCIOND DE PADRES
        if type == 'random':
            dx = {0:-1, 1:0, 2:1, 3:0}
            dy = {0:0, 1:-1, 2:0, 3:1}

            p2 = np.zeros((self.f,self.c,self.tamano_del_cromosoma))
            p2fit = np.zeros((self.f,self.c))

            for i in range(0,self.f):
                for j in range(0,self.c):
                    t = np.random.randint(0,4)
                    #print("t: {}\ndx[t]: {}\ndy[t]: {}".format(t,dx[t],dy[t]))
                    x = i + 1 + dx[t]
                    y = j + 1 + dy[t]
                    p2[i,j] = self.autEx[x,y]
                    p2fit[i,j] = self.fitEx[x,y]

            self.p2 = p2
            self.p2fit = p2fit

        if type == 'roulette':
            lts = [[-1,0],[0,1],[0,-1],[1,0]]#[-1,-1],[-1,1],[1,1],[1,-1]]
            self.p2 = np.zeros((self.f,self.c,self.tamano_del_cromosoma))
            self.p2fit = np.zeros((self.f,self.c))

            lista = []
            fitlista = []
            k=0
            for i in range(0,self.f):
                for j in range(0,self.c):
                    tee = []
                    teu = []
                    for m,n in lts:
                        tee.append(self.autEx[i + 1 + m, j + 1 + n])
                        teu.append(self.fitEx[i + 1 + m, j + 1 + n])
                    lista.append(tee)
                        #print(self.autEx[i + 1 + m, j + 1 + n])
                    fitlista.append(teu)
                    fitness = fitlista[k]
                    total_fitness = float(sum(fitness))
                    relative_fitness = [f/total_fitness for f in fitness]
                    self.__probability_value2 = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

                    for (w, individual) in enumerate(lista[k]):
                        r = random.random()
                        if r <= self.__probability_value2[w]:
                            self.p2[i,j] = individual
                            self.p2fit[i,j] = fitlista[k][w]
                            break
                    k+=1
    def mutation(self,chromosome):
        #"""NUEVOOOOOO"""
        if self.problem <= 6:
        #"""FINN NUEVOOOOOO"""
            r = random.random()
            occur = 0
            gen = []
            gen2 = []
            if r <= self.mutation_probability:

                k1=random.randint(0,len(chromosome)-1)
                if chromosome[k1]==0:
                    if self.mutation_type =="normal":
                        occur+=1
                        chromosome[k1]=1
                    if self.mutation_type == "evaluated":
                        ch2=[chromosome]
                        gen=Generation(ch2,self.problem)
                        chromosome[k1]=1
                        occur+=1
                        ch2=[chromosome]
                        gen2=Generation(ch2,self.problem)
                        if gen.get_individuals()[0].get_fitness() >= gen2.get_individuals()[0].get_fitness():
                            chromosome=gen.get_individuals()[0].chromo
                            occur-=1

                else:
                    if self.mutation_type == "normal":
                        occur+=1
                        chromosome[k1]=0
                    if self.mutation_type == "evaluated":
                        ch2=[chromosome]
                        gen=Generation(ch2,self.problem)
                        chromosome[k1]=0
                        occur+=1
                        ch2=[chromosome]
                        gen2=Generation(ch2,self.problem)
                        if gen.get_individuals()[0].get_fitness() >= gen2.get_individuals()[0].get_fitness():
                            chromosome=gen.get_individuals()[0].chromo
                            occur-=1

        #"""NUEVOOOOOO"""
        if (self.problem == 7) or (self.problem == 8):
            r = random.random()
            occur = 0
            gen = []
            gen2 = []
            if r <= self.mutation_probability:
                while(1):
                    k1=random.randint(0,len(chromosome)-1)
                    k2=random.randint(0,len(chromosome)-1)
                    if k1 != k2:
                        break

                if self.mutation_type == "normal-2permutation":
                    a1 = chromosome[k1]
                    a2 = chromosome[k2]

                    chromosome[k1] = a2
                    chromosome[k2] = a1

                    occur+=1

                if self.mutation_type == "evaluated-2permutation":
                    ch2=[chromosome]
                    gen=Generation(ch2,self.problem)
                    a1 = chromosome[k1]
                    a2 = chromosome[k2]
                    chromosome[k1] = a2
                    chromosome[k2] = a1
                    occur+=1
                    ch2=[chromosome]
                    gen2=Generation(ch2,self.problem)
                    if gen.get_individuals()[0].get_fitness() >= gen2.get_individuals()[0].get_fitness():
                        chromosome=gen.get_individuals()[0].chromo
                        occur-=1
                if self.mutation_type == "normal-npermutation":
                    if k1 < k2:
                        occur+=1
                        chromosome[k1:k2]=random.sample(chromosome[k1:k2],len(chromosome[k1:k2]))
                    if k1 > k2:
                        occur+=1
                        chromosome[k2:k1]=random.sample(chromosome[k2:k1],len(chromosome[k2:k1]))
                if self.mutation_type == "evaluated-npermutation":
                    ch2=[chromosome]
                    if self.problem == 8:
                        gff = 7
                    else:
                        gff = self.problem
                    gen=Generation(ch2,gff)
                    if k1 < k2:
                        occur+=1
                        chromosome[k1:k2]=random.sample(chromosome[k1:k2],len(chromosome[k1:k2]))
                    if k1 > k2:
                        occur+=1
                        chromosome[k2:k1]=random.sample(chromosome[k2:k1],len(chromosome[k2:k1]))
                    occur+=1
                    ch2=[chromosome]
                    if self.problem == 8:
                        gff = 7
                    else:
                        gff = self.problem
                    gen2=Generation(ch2,gff)
                    if gen.get_individuals()[0].get_fitness() >= gen2.get_individuals()[0].get_fitness():
                        chromosome=gen.get_individuals()[0].chromo
                        occur-=1

        #"""FINN NUEVOOOOOO"""
        return chromosome, occur

    def cross(self):
        count = 0
        self.__childs = []
        self.__mutations = 0
        occur=0
        while(1):
            self.__parents = []
            for n in range(0,2):
                r = random.random()
                for (i, individual) in enumerate(self.list_of_individuals.get_individuals()):
                    if r <= self.__probability_value[i]:
                        self.__parents.append(list(individual.chromo))
                        break
            r = random.random()
            if r <= self.cross_probability:
                #"""NUEVO"""
                if self.problem <=6:
                    self.__childs.append(self.__parents[0][:self.cross_point] + self.__parents[1][self.cross_point:])
                    self.__childs[count], occur= self.mutation(self.__childs[count])
                    self.__mutations+=occur
                    self.__childs.append(self.__parents[1][:self.cross_point] + self.__parents[0][self.cross_point:])
                    self.__childs[count+1], occur= self.mutation(self.__childs[count])
                    self.__mutations+=occur


                if self.problem==7:

                    p1=self.__parents[0][:self.cross_point]
                    k=p1.copy()
                    for i in range(0,len(self.__parents[0])):
                        try:
                            inx=p1.index(self.__parents[1][i])
                        except ValueError:
                            k.append(self.__parents[1][i])
                    self.__childs.append(k)

                    p2=self.__parents[1][:self.cross_point]
                    k=p2.copy()
                    for i in range(0,len(self.__parents[0])):
                        try:
                            inx=p2.index(self.__parents[0][i])
                        except ValueError:
                            k.append(self.__parents[0][i])

                    self.__childs.append(k)
                    self.__childs[count], occur = self.mutation(self.__childs[count])
                    self.__mutations += occur
                    self.__childs[count+1], occur = self.mutation(self.__childs[count+1])
                    self.__mutations += occur
                    """print("Padres({}):".format(self.cross_point) )
                    for i in range(0,len(self.__parents)):
                        print("{}".format(self.__parents[i]))
                    print("Hijos:")
                    for i in range(0,len(self.__childs)):
                        print("{}".format(self.__childs[i]))"""

                #"""FIN NUEVO
            #print("{} \t {}".format(len(self.__childs),self.list_of_individuals.get_individuals()))
            if len(self.__childs)==len(self.list_of_individuals.get_individuals()):
                if self.elitism == "direct-elitism":
                    self.__childs[random.randint(0,len(self.__childs)-1)] = self.list_of_individuals.get_second_best_individual().chromo
                    self.__childs[random.randint(0,len(self.__childs)-1)] = self.list_of_individuals.get_best_individual().chromo

                if self.elitism == "absolute-elitism":
                    ChildsAux=Generation(self.__childs,self.problem)
                    self.__childs[self.__childs.index(ChildsAux.get_second_worst_individual().chromo)] = self.list_of_individuals.get_second_best_individual().chromo
                    self.__childs[self.__childs.index(ChildsAux.get_worst_individual().chromo)] = self.list_of_individuals.get_best_individual().chromo
                break

    def sCross(self):
        self.problem = 8
        count = 0
        a = self.aut
        b = self.p2
        A = a.tolist()
        B = b.tolist()
        C = np.ones(np.shape(A))
        C = C.tolist()
        self.__childs = []
        self.__mutations = 0
        occur=0
        for i in range(0,np.shape(a)[0]):
            for j in range(0,np.shape(a)[1]):
                self.cross_point = random.randint(1,11)
                slc = random.choice([True, False])

                if slc :
                    p1=A[i][j][:self.cross_point]
                    k=p1.copy()
                    for t in range(0,len(A[i][j])):
                        try:
                            inx=p1.index(B[i][j][t])
                        except ValueError:
                            k.append(B[i][j][t])
                    self.__childs.append(k)
                else:
                    p2=B[i][j][:self.cross_point]
                    k=p2.copy()
                    for t in range(0,len(A[i][j])):
                        try:
                            inx=p2.index(A[i][j][t])
                        except ValueError:
                            k.append(A[i][j][t])
                    self.__childs.append(k)
                self.__childs[count], occur = self.mutation(self.__childs[count])
                self.__mutations += occur
                #print("LLEGA")

                C[i][j] = self.__childs[count]
                count += 1




        chromoc=C[i][j].copy()
        chromoc.append(chromoc[0])
        posiciones = {1:[0,0],2:[0,1],3:[0,2],4:[0,3],
        5:[1,3],6:[2,3],7:[3,3],8:[3,2],
        9:[3,1],10:[3,0],11:[2,0],12:[1,0]}
        s=0
        for i in range(1,len(chromoc)):
            s+=self.distancia(posiciones[chromoc[i-1]],posiciones[chromoc[i]])
        self.__fitnessCC = (120)/s

        i,j = np.where(self.fit == self.fit.max())
        i = int(i[0])
        j = int(j[0])
        if self.__fitnessCC<self.fit[i,j]:
            C[i][j] = self.aut[i,j]

        return C, self.__mutations

#Cruce de dos puntos
    def distancia(self,A,B):
        d=math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
        return d

    def kCross(self):
        pass

    def get_childs(self):
        return self.__childs

    def get_mutation(self):
        return self.__mutations

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
    return B

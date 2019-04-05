#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Modulo Creado por Carlos E. Gonzalez C.
#Para la materia Algoritmos Geneticos

import math

class ChromosomeClass():
    """docstring for ChromosomeClass."""
    __fitness = 0
    __size_of_chromo = 0

    def __init__(self, chromo,problem,gray=False):
        #fitness calculation
        self.chromo = chromo
        #sizeOfChromo calculation
        self.__size_of_chromo=len(self.chromo)

        if problem == 1:
            sum=0
            for i in range(0,len(self.chromo)):
                sum += self.chromo[i]
            self.__fitness = sum

        if problem == 2:
            zero = False
            sum=0
            for i in range(0,len(self.chromo)):
                if i==0:
                    if chromo[i] == 0:
                        zero = True
                    else:
                        zero = False
                else:
                    if zero and (i%2==1):
                        if chromo[i] == 1:
                            sum+=1
                    if zero and (i%2==0):
                        if chromo[i] == 0:
                            sum+=1
                    if (not zero) and (i%2==1):
                        if chromo[i] == 0:
                            sum+=1
                    if (not zero) and (i%2==0):
                        if chromo[i] == 1:
                            sum+=1
            self.__fitness=sum + 1

        if problem == 3:
            if gray:
                self.chromo=g2b(self.chromo)
                z=self.b2z(self.chromo)
                r=self.z2r(z,3.14159265359,0,16)
            else:
                z=self.b2z(self.chromo)
                r=self.z2r(z,3.14159265359,0,16)
            self.__fitness=math.sin(r)

        if problem == 4:
            if gray:
                self.chromo=g2b(self.chromo)
                z=self.b2z(self.chromo)
                r=self.z2r(z,(2*3.14159265359),0,16)
            else:
                z=self.b2z(self.chromo)
                r=self.z2r(z,(2*3.14159265359),0,16)
            self.__fitness=(math.sin(r))**2

        if problem == 5:
            if gray:
                self.chromo=g2b(self.chromo)
                z=self.b2z(self.chromo)
                r=self.z2r(z,(3.14159265359/2),0,16)
            else:
                z=self.b2z(self.chromo)
                r=self.z2r(z,(3.14159265359/2),0,16)
            self.__fitness=math.sin(r)*math.cos(r)

        if problem == 6:
            if gray:
                pass
            else:
                x=self.b2z(self.chromo[0:16])
                y=self.b2z(self.chromo[16:32])
                z=self.b2z(self.chromo[32:48])
                x=self.z2r(x,(3.14159265359/2),(-3.14159265359/2),48)
                y=self.z2r(y,(3.14159265359/2),(-3.14159265359/2),48)
                z=self.z2r(z,(3.14159265359/2),(-3.14159265359/2),48)
            self.__fitness=((math.sin(x))*(math.cos(y))*(math.tan(z)))

        #"""CODIGO NUEVOOOOOO"""
        if problem == 7:
            chromoc=chromo.copy()
            chromoc.append(chromo[0])
            posiciones = {1:[0,0],2:[0,1],3:[0,2],4:[0,3],
            5:[1,3],6:[2,3],7:[3,3],8:[3,2],
            9:[3,1],10:[3,0],11:[2,0],12:[1,0]}
            s=0
            for i in range(1,len(chromoc)):
                s+=self.distancia(posiciones[chromoc[i-1]],posiciones[chromoc[i]])

            self.__fitness = 12/s
        #"""FIN CODIGO NUEVOOOOOO"""

    def get_fitness(self):
        return self.__fitness

    def get_size_of_chromo(self):
        return self.__size_of_chromo

    def b2g(self,b):
        z=b2z(b)
        z ^= (z >> 1)
        g=[int(x) for x in str(bin(z)[2:])]
        return g

    def g2b(self,g):
        g="".join(str(x) for x in g)
        g = int(g, 2)
        mask = g
        while mask != 0:
            mask >>= 1
            g ^= mask
        b=bin(g)[2:]
        b=[int(x) for x in str(b)]
        return b

    def z2r(self,z,rmax,rmin,l):
        r=(((rmax-rmin)*z)/(2**(l)-1))+rmin
        return r

    def b2z(self,b):
        b="".join(str(x) for x in b)
        z=int(b,2)
        return z

    #"""CODIGO NUEVOOOOOO"""
    def distancia(self,A,B):
        d=math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
        return d
    #"""FIN CODIGO NUEVOOOOOO"""

import numpy as np

def extendedMatrix(m):
    s=np.shape(m)
    B=[]
    A=m
    if len(s)!=3:
        print("Error, la matriz debe ser al menos de 2 dimensiones y maximo de 3 demensioens.")
    else:
        B=np.pad(A,((1, 1), (1, 1), (0,0)), 'wrap')
        B[0,0]=0
        B[0,s[1]+1]=0
        B[s[0]+1,0]=0
        B[s[0]+1,s[1]+1]=0
    return B

t=10*np.random.rand(3,3,1)
print("ORIGINAL")
print(t)
print(np.shape(t))
print("EXTENDIDA")

print(extendedMatrix(t))
print(np.reshape(extendedMatrix(t),(3,3)))

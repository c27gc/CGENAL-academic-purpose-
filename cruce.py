def sCross(self, a, b):
    self.problem = 8
    count = 0
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

    return C, self.__mutations

def kCross():
    

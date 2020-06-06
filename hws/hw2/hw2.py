from scipy.optimize import fsolve,leastsq
import numpy as np



class TD_lambda:
    def __init__(self, probToState,valueEstimates,rewards):
        self.probToState = probToState
        self.valueEstimates = valueEstimates
        self.rewards = rewards
        self.td1 = self.get_vs0(1)

    def get_vs0(self,lambda_):
        probToState = self.probToState
        valueEstimates = self.valueEstimates
        rewards = self.rewards
        vs = dict(zip(['vs0','vs1','vs2','vs3','vs4','vs5','vs6'],list(valueEstimates)))

        vs5 = vs['vs5'] + 1*(rewards[6]+1*vs['vs6']-vs['vs5'])
        vs4 = vs['vs4'] + 1*(rewards[5]+lambda_*rewards[6]+lambda_*vs['vs6']+(1-lambda_)*vs['vs5']-vs['vs4'])
        vs3 = vs['vs3'] + 1*(rewards[4]+lambda_*rewards[5]+lambda_**2*rewards[6]+lambda_**2*vs['vs6']+lambda_*(1-lambda_)*vs['vs5']+(1-lambda_)*vs['vs4']-vs['vs3'])
        vs1 = vs['vs1'] + 1*(rewards[2]+lambda_*rewards[4]+lambda_**2*rewards[5]+lambda_**3*rewards[6]+lambda_**3*vs['vs6']+lambda_**2*(1-lambda_)*vs['vs5']+lambda_*(1-lambda_)*vs['vs4']+\
                            (1-lambda_)*vs['vs3']-vs['vs1'])
        vs2 = vs['vs2'] + 1*(rewards[3]+lambda_*rewards[4]+lambda_**2*rewards[5]+lambda_**3*rewards[6]+lambda_**3*vs['vs6']+lambda_**2*(1-lambda_)*vs['vs5']+lambda_*(1-lambda_)*vs['vs4']+\
                            (1-lambda_)*vs['vs3']-vs['vs2'])

        vs0 = vs['vs0'] + probToState*(rewards[0]+lambda_*rewards[2]+lambda_**2*rewards[4]+lambda_**3*rewards[5]+lambda_**4*rewards[6]+lambda_**4*vs['vs6']+lambda_**3*(1-lambda_)*vs['vs5']+\
                                    +lambda_**2*(1-lambda_)*vs['vs4']+lambda_*(1-lambda_)*vs['vs3']+(1-lambda_)*vs['vs1']-vs['vs0']) +\
                (1-probToState)*(rewards[1]+lambda_*rewards[3]+lambda_**2*rewards[4]+lambda_**3*rewards[5]+lambda_**4*rewards[6]+lambda_**4*vs['vs6']+lambda_**3*(1-lambda_)*vs['vs5']+\
                                    +lambda_**2*(1-lambda_)*vs['vs4']+lambda_*(1-lambda_)*vs['vs3']+(1-lambda_)*vs['vs2']-vs['vs0'])
        return vs0

    def get_lambda(self,x0=np.linspace(0.1,1,10)):
        return fsolve(lambda lambda_:self.get_vs0(lambda_)-self.td1, x0)

def helper(p,V,R):
    tdl = TD_lambda(p, V, R)
    return tdl.get_lambda()

def ex1():
    p=0.81
    V=[0.0,4.0,25.7,0.0,20.1,12.2,0.0]
    R=[7.9,-5.1,2.5,-7.2,9.0,0.0,1.6]

    print("ANSWER=",helper(p=p,V=V,R=R))
#ex1()


def ex2():
    p = 0.22
    V = [12.3, -5.2, 0.0, 25.4, 10.6, 9.2, 0.0]
    R = [-2.4, 0.8, 4.0, 2.5, 8.6, -6.4, 6.1]
    print("ANSWER=",helper(p=p,V=V,R=R))
#ex2()


def ex3():
    p = 0.64
    V = [-6.5, 4.9, 7.8, -2.3, 25.5, -10.2, 0.0]
    R = [-2.4, 9.6, -7.8, 0.1, 3.4, -2.1, 7.9]
    print("ANSWER=",helper(p=p,V=V,R=R))
#ex3()


###########
def q1():
    p = 0.92

    V = [0.0, 13.1, 1.3, 24.5, 19.9, 21.5, 0.0]

    R = [5.3, 0.6, 6.7, -2.2, 8.6, 4.1, -3.2]

    print("ANSWER=",helper(p=p,V=V,R=R))
q1()


def q2():
    p = 0.32

    V = [6.5, 16.8, 0, 24, 17.3, -4.6, 0.0]

    R = [3.2, 0, -0.3, 7.6, 4.1, 4.7, 0.8]

    print("ANSWER=",helper(p=p,V=V,R=R))
q2()


def q3():
    p = 0.97

    V = [19.4, 6.6, 2.2, 0, 23.3, -1.8, 0.0]

    R = [4.6, 6.2, 0, 3.2, 5.4, -2.5, 4.2]

    print("ANSWER=",helper(p=p,V=V,R=R))
q3()


def q4():
    p = 0.61

    V = [-3.2, 0, -3.5, 19.9, 24.2, 21.4, 0.0]

    R = [7.5, 9.6, 1.6, 2.9, -1.2, 9.5, -2.3]
    print("ANSWER=",helper(p=p,V=V,R=R))
q4()


def q5():
    p = 0.09

    V = [19.0, 0, 14, 4.7, 7.9, 13.8, 0.0]

    R = [-0.4, -1.1, 9.7, 7.4, 1.6, 0.5, 3.6]

    print("ANSWER=",helper(p=p,V=V,R=R))
q5()


def q6():
    p = 0.64

    V = [0.0, 0, 6.8, 13.2, -4, 14.8, 0.0]

    R = [8.5, 0, 7.9, -0.5, 8.6, -4.1, -2.4]

    print("ANSWER=",helper(p=p,V=V,R=R))
q6()


def q7():
    p = 0.63

    V = [13.3, 0, -1.5, 0, 12.4, 7.9, 0.0]

    R = [-3.0, -3.5, 4.8, 8.7, 0, -1.7, -1.3]

    print("ANSWER=",helper(p=p,V=V,R=R))
q7()


def q8():
    p = 1.0

    V = [13.1, 7.6, 7.5, 21.4, 12.1, 11.7, 0.0]

    R = [-1.6, 0.2, 2.8, -1.2, 4.1, 9, -4.7]

    print("ANSWER=",helper(p=p,V=V,R=R))
q8()



def q9():
    p = 0.0

    V = [21.9, -1.1, 0, 23.2, 12.8, 0, 0.0]

    R = [1.0, 4.5, 0, 2.8, 3.5, 2.5, 1.2]

    print("ANSWER=",helper(p=p,V=V,R=R))
q9()


def q10():
    p = 0.0

    V = [0.0, 20.9, 8.9, 0.7, 0, 10, 0.0]

    R = [5.1, 8.3, -2, 0.8, 0, 1.3, 6.2]

    print("ANSWER=",helper(p=p,V=V,R=R))
q10()
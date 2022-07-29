import math
import random
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from timeit import default_timer
from numpy import unique, ravel
from numpy import sqrt, array, diagonal, mean, transpose, eye, ones
from numpy import transpose, diag, dot
from numpy.linalg import svd, inv, qr, det
from sklearn.linear_model import LinearRegression


def topTeams(__PRs, __U, __n = 10, TR=False):
    dd = {i: __PRs[i] for i in range(len(__PRs))}
    dd = {__U[team]: rank for team, rank in sorted(dd.items(), key=lambda kv: kv[1], reverse=True)}
    __teams = list(dd.keys())[:__n]
    dd= {page: dd[team] for team in __teams}
    if not TR:
        return dd
    return {"Link": list(dd.keys()), "Value": ravel(list(dd.values()))}


def viewTopTeams(__PRs, __U, __n = 10):
    if type(__PRs[0]) in (list, np.array, np.matrix):
        __PRs = np.ravel(__PRs)
    __topTeams = topTeams(__PRs, __U, __n)
    for __team in __topTeams:
        print(__team, __topTeams[__page])
    return __topTeams
    

def convertToP(_A, tp=0.85):
    n = len(_A)
    ntp = (1-tp)/n
    ninv = 1/n
    P = []
    for i in range(n):
        r = []
        ri = np.sum(_A[i])
        if np.sum(_A[i]) > 0:
            for j in range(n):
                r.append(tp*(_A[i,j]/ri) + ntp)
        else:
            for j in range(n):
                r.append(ninv)
        P.append(r)
    return array(P)


def topTeamsFromMtx(__A, tp=0.85):
    P = convertToP(__A, tp)
    IP = np.eye(n) - P.T
    b = np.zeros(n).reshape(n,1)
    IP[0] = ones(n)
    b[0] = 1.0
    IP[0] = ones(n)
    u,d,v = svd(IP)
    x = dot(v.T, inv(diag(d))).dot(u.T).dot(b)
    viewTopTeams(x, team_names)


def topTeams2(__A, tp=0.85,cc=10):
    P = convertToP(__A, tp)
    IP = np.eye(n) - P.T
    b = np.zeros(n).reshape(n,1)
    IP[0] = ones(n)
    b[0] = 1.0
    IP[0] = ones(n)
    u,d,v = svd(IP)
    x = dot(v.T, inv(diag(d))).dot(u.T).dot(b)
    return topTeams(x, team_names, cc)


# Optional: using the iterative page algorithm solver
class Page(object):
    def __init__(self, index):
        self.index = index
        self.in_links = []
        self.out_links = []
        self.rank = 0.01


class Network(object):
    def setupLinks(self):
        for page in self.pages:
            for i in range(len(self.A)):
                if self.A[page.index, i] > 0:
                    page.out_links.append(self.pages[i])
                if self.A[i, page.index] > 0:
                    page.in_links.append(self.pages[i])
    
    
    def __init__(self, adj_mtx):
        self.A = adj_mtx
        self.pages = [Page(index) for index in range(len(adj_mtx))]
        self.setupLinks()


    def surf(self, max_iter):
        for itr in range(max_iter):
            for page in self.pages:
                page.rank = TP * sum([p.rank / len(p.out_links) for p in page.in_links]) + TP_away


    def getPageRanks(self):
        norm = 1 / sum([page.rank for page in self.pages])
        return [norm * page.rank for page in self.pages]


def makeB(_A, p):
    B = np.zeros((p,p))
    B = np.matrix(B)
    for i in range(p):
        for j in range(p):
            if i != j:
                c1 = _A[i, j]
                c2 = _A[j, i]
                if c2 > c1 and c1 != 0:
                    B[j,i] = c2 - c1
                elif c1 > c2 and c2 != 0:
                    B[i,j] = c1 - c2
    return B

# Binary Matrix
def makeC(_A, p):
    C = np.zeros((p,p))
    C = np.matrix(C)
    for i in range(p):
        for j in range(p):
            if i != j:
                c1 = _A[i, j]
                c2 = _A[j, i]
                if c2 > c1 and c1 != 0:
                    C[j,i] = 1
                elif c1 > c2 and c2 != 0:
                    C[i,j] = 1
    return C


def makeD(_A, p):
    D = np.zeros((p,p))
    D = np.matrix(C)
    for i in range(len(_A)):
        for j in range(len(_A[0])):
            rs = np.sum(_A[i])
            cs = np.sum(_A[:, j])
            ts = rs + cs - _A[i,j]
            D[i,j] =  _A[i,j]/ts
    return D


# p = 30 (the number of teams)
p = len(team_names)

# A = pxp non-symetric adjacency matrix containing
# data for the games of the MLB 2021 season
A = np.zeros((p,p))
A = np.matrix(A)

# if team 1 won, then team 2 feeds/adds "dominance" to team 1
for i in range(1000,len(df)-50):
    if df.iloc[i]["WinningTeam"] == 1:
        ind_i = team_names.index(df.iloc[i]["Team2"])
        ind_j = team_names.index(df.iloc[i]["Team1"])
        A[ind_i, ind_j] += 1
    # vice versa
    if df.iloc[i]["WinningTeam"] == 2:
        ind_i = team_names.index(df.iloc[i]["Team1"])
        ind_j = team_names.index(df.iloc[i]["Team2"])
        A[ind_i, ind_j] += 1


# Alternative matrices to A
B = makeB(A, p)
C = makeC(A, p)
D = makeD(A, p)

# Optional: using the iterative page algorithm solver
# note that A vs A.T should be the best and the worst teams
"""
n = len(A)
ninv = 1/n
TP = 0.85
TP_away = (1-TP)
network = Network(A)
network.surf(150)
"""


A_rnorm = A.copy()
for i in range(len(A_rnorm)):
    A_rnorm[i] = A_rnorm[i]/np.sum(A_rnorm[i])


A_cnorm = A.copy()
for j in range(len(A_cnorm)):
    A_cnorm[:,j] = A_cnorm[:,j]/np.sum(A_cnorm[:,j])


# A dictionary of The Team Ranks
ttA = topTeams2(A, 0.85, 11)


# A seaborn barplot of the Team Ranks (Jupyter Environment)
"""
sns.barplot(y="Team", x="Dominance", data=pd.DataFrame({ "Dominance":ravel(list(ttA.values())), "Team": list(ttA.keys()) }))
"""



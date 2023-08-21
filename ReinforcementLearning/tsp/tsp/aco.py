# -*- coding: utf-8 -*-
"""
Ant Colony Algorithm for TSP
"""

import pandas, math,time
import matplotlib.pyplot as plt
import numpy as np

class Node:
    """
        class Name：Node
        info：	city node
    """

    def __init__(self, CityNum):

        self.visited = [False] * CityNum  # Record whether cities have been visited.
        self.start = 0  # Starting city.
        self.end = 0  # Destination city.
        self.current = 0  # Current city
        self.num = 0  # Number of cities visited.
        self.pathsum = 0  # Total distance traveled.
        self.lb = 0  # Lower bound of the current node
        self.listc = []  # Record the sequence of visited cities.


def GetData(datapath):
    """
    outputs:
        Position: Matrix of positions of various cities.
        CityNum: Number of cities.
        Dist: Matrix of distances between cities.
    """
    dataframe = pandas.read_csv(datapath, sep=" ", header=None)
    Cities = dataframe.iloc[:, 1:3]
    Position = np.array(Cities)
    CityNum = Position.shape[0]
    Dist = np.zeros((CityNum, CityNum))

    # Calculate the distance matrix.
    for i in range(CityNum):
        for j in range(CityNum):
            if i == j:
                Dist[i, j] = math.inf
            else:
                Dist[i, j] = math.sqrt(np.sum((Position[i, :] - Position[j, :]) ** 2))
    return Position, CityNum, Dist


def ResultShow(Min_Path, BestPath, CityNum, string):

    print("The obtained shortest path for the traveling salesman is：")
    for m in range(CityNum):
        print(str(BestPath[m]) + "—>", end="")
    print(BestPath[CityNum])
    print("The total path length is：" + str(Min_Path))
    print()


def draw(BestPath, Position, title):
    plt.title(title)
    plt.plot(Position[:, 0], Position[:, 1], 'bo')
    for i, city in enumerate(Position):
        plt.text(city[0], city[1], str(i))
    plt.plot(Position[BestPath, 0], Position[BestPath, 1], color='red')
    plt.show()

def ant():

    numant = 25  # Number of ants.
    numcity = CityNum  # Number of cities
    alpha = 1  # Importance factor of pheromone.
    rho = 0.1  # Pheromone evaporation rate.
    Q = 1

    iters = 0
    itermax = 500
    #Heuristic function matrix represents the expected degree of ants transferring from city i to city j.
    etatable = 1.0 / (Dist + np.diag([1e10] * numcity))
    pheromonetable = np.ones((numcity, numcity))
    pathtable = np.zeros((numant, numcity)).astype(int)
    # Average length of paths for each generation.
    lengthaver = np.zeros(itermax)
    #The best path length encountered by each generation and its predecessors.
    lengthbest = np.zeros(itermax)
    #The best path encountered by each generation and its predecessors.
    pathbest = np.zeros((itermax, numcity))

    while iters < itermax:
        #  Randomly generate the starting city for each ant.
        if numant <= numcity:
            #The number of cities is greater than the number of ants.
            pathtable[:, 0] = np.random.permutation(range(0, numcity))[:numant]
        else:
            # There are more ants than cities, additional ants need to be assigned.
            pathtable[:numcity, 0] = np.random.permutation(range(0, numcity))[:]
            pathtable[numcity:, 0] = np.random.permutation(range(0, numcity))[:numant - numcity]
        # Calculate the path distance for each ant.
        length = np.zeros(numant)

        for i in range(numant):
            visiting = pathtable[i, 0]


            unvisited = set(range(numcity))
            unvisited.remove(visiting)

            for j in range(1, numcity):
                # Use roulette wheel selection to choose the next city to be visited each time.
                listunvisited = list(unvisited)
                probtrans = np.zeros(len(listunvisited))

                for k in range(len(listunvisited)):
                    probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]], alpha) \
                                   * np.power(etatable[visiting][listunvisited[k]], alpha)
                cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
                cumsumprobtrans -= np.random.rand()

                k = listunvisited[np.where(cumsumprobtrans > 0)[0][0]]  # The next city to be visited.
                pathtable[i, j] = k
                unvisited.remove(k)
                # visited.add(k)
                length[i] += Dist[visiting][k]
                visiting = k
            # The path distance of an ant includes the distance between the last city and the first city.
            length[i] += Dist[visiting][pathtable[i, 0]]

        # After completing an iteration that includes all ants, calculate several statistical parameters for this iteration.
        lengthaver[iters] = length.mean()
        if iters == 0:
            lengthbest[iters] = length.min()
            pathbest[iters] = pathtable[length.argmin()].copy()
        else:
            if length.min() > lengthbest[iters - 1]:
                lengthbest[iters] = lengthbest[iters - 1]
                pathbest[iters] = pathbest[iters - 1].copy()
            else:
                lengthbest[iters] = length.min()
                pathbest[iters] = pathtable[length.argmin()].copy()

            # update heromone
        changepheromonetable = np.zeros((numcity, numcity))
        for i in range(numant):
            for j in range(numcity - 1):
                changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / length[i]
            changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / length[i]
        pheromonetable = (1 - rho) * pheromonetable + changepheromonetable

        iters += 1

    path_tmp = pathbest[-1]
    BestPath = []
    for i in path_tmp:
        BestPath.append(int(i))
    BestPath.append(BestPath[0])

    return BestPath, lengthbest[-1]



if __name__ == "__main__":
    Position, CityNum, Dist = GetData("/home/xinao/workplace/git/rl_notes/ReinforcementLearning/tsp/TSP25cities.tsp")

    start = time.time()
    BestPath, Min_Path = ant()
    end = time.time()

    print(end-start)
    ResultShow(Min_Path, BestPath, CityNum, "ACO")
    draw(BestPath, Position, "Ant Method")
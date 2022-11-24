import random
import numpy as np

Dimension = 8
NumMines = 6


def GetNeighbours(Position):
    NeighboursList = [[x, y] for x in range(Position[0] - 1, Position[0]+2)
                             for y in range(Position[1] - 1, Position[1]+2)]
    NeighboursOut = []
    for Neighbour in NeighboursList:
        if Neighbour[0] < 0 or Neighbour[1] < 0 or Neighbour[0] > (Dimension - 1) or Neighbour[1] > (Dimension - 1):
            NeighboursOut.append(Neighbour)
    NeighboursList.remove(Position) 
    return [x for x in NeighboursList if x not in NeighboursOut]  

def RandomMines(Dimension, NumMines):
    MinesPos = []
    while (len(MinesPos) != NumMines):
        Mine = [random.randint(0, Dimension - 1), random.randint(0, Dimension - 1)]
        if Mine in MinesPos:
            continue
        else:
            MinesPos.append(Mine)
    return MinesPos


def ConstructNet (Dimension, NumMines):

    MinesPos = RandomMines(Dimension, NumMines)
    Net = np.zeros((Dimension,Dimension))
    for Mine in MinesPos:
        neighbours = GetNeighbours(Mine)
        Net[Mine[0], Mine[1]] = 12
        for neighbour in neighbours:
            if neighbour in MinesPos:
                continue
            Net[neighbour[0], neighbour[1]] += 1   
    return Net, MinesPos


def PrintNet(Net):
    print('┌ ', end="", flush=True)
    for x in range(Dimension):
        print(' '+ str(x)+ ' ', end="", flush=True)
    print(' ┐')
    y = 0
    for tile in Net:
        print( str(y) +'  ',end="", flush=True)
        for element in tile:
            if element == 10: # undiscovered tile
                print('   ',end="", flush=True)
            elif element == 11: # flag
                print('>  ',end="", flush=True)
            elif element == 12: # landmine
                print('■  ',end="", flush=True)
            else: print(''+ str(int(element))+ '  ',end="", flush=True)
        y += 1   
        print('|')
    print('└ ', end="", flush=True)

    for x in range(Dimension):
        print(' - ', end="", flush=True)
    print(' ┘')

    return None
'''
def FindOtherZeros(Net, PickedLocation):
    Searching = True
    while Searching == True:
        Possible = GetNeighbours(Position)
        for 
'''
def RevealAroundZero(Zeros):
    revealed = []
    for Zero in Zeros:
        LocalNeighbours = GetNeighbours(Zero)
        for LocalNeighbour in LocalNeighbours:
            if LocalNeighbour in revealed:
                continue
            revealed.append(LocalNeighbour)
        #print(revealed)
    return revealed

def StartGame(Dimension, NumMines):
    FoundMines = 0
    AlredyPicked = []
    Net, MinesPos = ConstructNet(Dimension, NumMines) 
    NetKnown = np.full((Dimension,Dimension), 10)
    #PrintNet(Net)


    while FoundMines != NumMines:
        PrintNet(NetKnown)
        print('Would you like to place a flag? [y,n]')
        flagging = input()
        print('Pick a spot! (format: \"x,y\", e.g. for x=0 and y=3 write "0,3")')
        inp1 = input()
        PickedLocation = [int(inp1[2]), int(inp1[0])]
        if PickedLocation in AlredyPicked:
            print('This tile was already revealed!')
            continue
        else: 
            AlredyPicked.append(PickedLocation)   
        if flagging == 'y':
            NetKnown[PickedLocation[0], PickedLocation[1]] = 11
            if PickedLocation in MinesPos:
                FoundMines += 1
        else:
            if PickedLocation in MinesPos:
                print('You struck a mine! GAME OVER :(')
                PrintNet(Net)
                return None
            else:
                NewTile = Net[PickedLocation[0], PickedLocation[1]]
                if NewTile == 0:
                    Zeros = []
                    for i in range(Dimension):
                        for j in range(Dimension):
                            if Net[i,j] == 0:
                                Zeros.append([i,j])
                    for Zero in Zeros:
                        NetKnown[Zero[0], Zero[1]] = 0
                    Revealed = RevealAroundZero(Zeros)
                    for element in Revealed:
                        NetKnown[element[0], element[1]] = Net[element[0], element[1]]
                else:
                    NetKnown[PickedLocation[0], PickedLocation[1]] = NewTile

    print('You are victorious!')
    PrintNet(Net)
    return None



StartGame(Dimension, NumMines)



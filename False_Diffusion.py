import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from TDMA import LBL_TDMA

linestyle = [
    ((0, (1, 1))),
    ((5, (10, 3))),
    ((0, (5, 10))),
    ((0, (5, 5))),
    ((0, (3, 10, 1, 10))),
    ((0, (3, 5, 1, 5))),
    ((0, (3, 5, 1, 5, 1, 5))),
    ((0, (3, 10, 1, 10, 1, 10))),
    ((0, (3, 1, 1, 1, 1, 1)))]

gridNumber = [10, 50, 100, 200]

thickness = 0.1
plateLength = 1
plateWidth = 1

tempLeft = 100
tempTop = 100
tempBottom = 0
tempRight = 0

for j, i in enumerate(gridNumber):
    nCellsLength = i
    nCellsWidth = i

    CellLength = plateLength / nCellsLength
    CellWidth = plateWidth / nCellsWidth
    CellVolume = CellLength * CellWidth * thickness

    rho = 1
    ux = 1
    uv = 1
    Fx = np.zeros((nCellsWidth, nCellsLength))
    Fv = np.zeros((nCellsWidth, nCellsLength))

    Fx += rho * ux
    Fv += rho * uv

    xC = np.array(
        [CellLength / 2 + i * CellLength for i in range(nCellsLength)])
    yC = np.array([CellWidth / 2 + i * CellWidth for i in range(nCellsWidth)])

    xF = np.linspace(0, plateLength, nCellsLength+1)
    yF = np.linspace(0, plateWidth, nCellsWidth+1)

    xCentroid, yCentroid = np.meshgrid(xC, yC)
    xFace, yFace = np.meshgrid(xF, yF)

    Coefficient_Matrix = np.ones((nCellsWidth, nCellsLength))
    WestBoundary = np.zeros_like(Coefficient_Matrix)
    WestBoundary[:, 0] = 1

    EastBoundary = np.zeros_like(Coefficient_Matrix)
    EastBoundary[:, -1] = 1

    NorthBoundary = np.zeros_like(Coefficient_Matrix)
    NorthBoundary[0, :] = 1

    SouthBoundary = np.zeros_like(Coefficient_Matrix)
    SouthBoundary[-1, :] = 1

    aW = Fx * np.multiply(np.ones_like(Coefficient_Matrix), 1-WestBoundary)
    aE = Fx * np.zeros((nCellsWidth, nCellsLength))
    aS = Fv * np.multiply(np.ones_like(Coefficient_Matrix), 1-SouthBoundary)
    aN = Fv * np.zeros((nCellsWidth, nCellsLength))
    aP = np.multiply(Coefficient_Matrix, Fx+Fv)

    Su = np.zeros_like(Coefficient_Matrix)
    Su = np.multiply(np.ones_like(Coefficient_Matrix), WestBoundary)*tempLeft
    Su += np.multiply(np.ones_like(Coefficient_Matrix),
                      SouthBoundary)*tempBottom

    result = LBL_TDMA(aP, aN, aS, aW, aE, Su)

    result = np.vstack([tempTop*np.ones((1, nCellsLength)),
                       result, tempBottom*np.ones((1, nCellsLength))])
    result = np.hstack([tempLeft*np.ones((nCellsWidth+2, 1)),
                       result, tempRight*np.ones((nCellsWidth+2, 1))])

    xNode = np.hstack([xFace[:-1, 0].reshape(nCellsWidth, 1),
                      xCentroid, xFace[:-1, -1].reshape(nCellsWidth, 1)])
    xNodes = np.vstack([xNode[0, :], xNode, xNode[-1, :]])
    yNode = np.vstack([yFace[0, :-1], yCentroid, yFace[-1, :-1]])
    yNodes = np.flipud(np.hstack([yNode[:, 0].reshape(
        nCellsWidth+2, 1), yNode[:, :], yNode[:, -1].reshape(nCellsWidth+2, 1)]))

    plt.rcParams["figure.figsize"] = (6, 6)
    cmap_reversed = cm.get_cmap('autumn_r')

    plt.rcParams["figure.figsize"] = (8, 6)

    Tsimulation = np.diag(result)
    label = r"Upwind {}x{}".format(nCellsLength, nCellsWidth)
    plt.plot(xNodes[0], Tsimulation, label=label, linestyle=linestyle[j])

plt.xlabel(r'Distance Alonside Diagonal [m]', fontsize=11)
plt.ylabel(r'$Temperature$ [°C]', fontsize=11)
x = np.linspace(0, plateLength, 100)
Ttheorical = np.ones_like(x) * tempLeft
Ttheorical[int(len(x)/2):] = tempRight
plt.plot(x, Ttheorical, label="Exact Solution")
plt.legend()
plt.savefig("False Diffusion UD scheme.png")
plt.show()

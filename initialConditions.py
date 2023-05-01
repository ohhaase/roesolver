import json

varDict = {}

varDict["simNumber"] = "3"

varDict["simtype"] = "freestream"
varDict["limtype"] = "superbee"
varDict["xBoundConds"] = "periodic"
varDict["yBoundConds"] = "outflow"

varDict["saveInterval"] = 100

varDict["ax"] = 0
varDict["bx"] = 5
varDict["Nx"] = 100 

varDict["ay"] = 0
varDict["by"] = 5
varDict["Ny"] = 100

varDict["maxdt"] = 5
varDict["Nt"] = 100000
varDict["ti"] = 0

varDict["timelimit"] = 60

varDict["Cp"] = 1
varDict["Cv"] = 0.718

varDict["nGhost"] = 2 

varDict["Vinf"] = 100/2
varDict["rhoinf"] = 1.293*2
varDict["Pinf"] = 101000

fileName = "Roe2D_" + varDict["simtype"] + "_initConds_" + varDict["simNumber"] + ".json"

with open(fileName, 'w') as f:
    json.dump(varDict, f)
    f.close()
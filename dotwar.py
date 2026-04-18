# usage: dotwar.py universe [orders] time
# take in dotwar system as text file, run, output event log
# optionally take in order list from text file
from dotclass import *

args = sys.argv
universe = args[1]
arg2 = args[2]
try:
    endtime = int(arg2)
    orders = None
except ValueError:
    orders = arg2
    endtime = int(args[3])

entities = []
with open(universe, 'r') as universe_file:
    for ln in universe_file.readlines():
        if ln.startswith("#"): continue
        # name kind r r r v v v a a a allegiance [capabilities ...]
        ename, ekind, er, ev, ea, eallegiance, ecaps = None, None, np.array(()), np.array(()), np.array(()), None, {}
        tokens = ln.split()
        consttokens = tokens[0:12]
        ename, ekind = consttokens[:2]
        er = np.array([float(r) for r in consttokens[2:2+3]])
        ev = np.array([float(v) for v in consttokens[5:5+3]])
        ea = np.array([float(a) for a in consttokens[8:8+3]])
        print(ev)
        eallegiance = consttokens[9]
        captokens = [token.upper() for token in tokens[10:]]
        capindex = 0
        while capindex < len(captokens):
            captoken = captokens[capindex]
            skip = 1
            names = ['ENGINE', 'TANK', 'BAY', 'REFINE', 'DETONATE']
            if captoken == 'ENGINE':
                ecaps[Capability.ENGINE] = {}
            elif captoken == 'REFINE':
                ecaps[Capability.REFINE] = {}
            elif captoken == 'DETONATE':
                ecaps[Capability.DETONATE] = {}
            elif captoken == 'TANK':
                tcurrent = int(captokens[capindex+1])
                tmax = int(captokens[capindex + 2])
                ecaps[Capability.TANK] = {"current": tcurrent, "max": tmax}
                skip = 3
            elif captoken == 'BAY':
                tcurrent = int(captokens[capindex + 1])
                tmax = int(captokens[capindex + 2])
                ecaps[Capability.BAY] = {"current": tcurrent, "max": tmax}
                skip = 3
            capindex += skip
        e = Entity(ename, ekind, er, ev, ea, eallegiance, ecaps)
        entities.append(e)

sim = Simulation(0, entities, [])
sim.run(endtime)
print("total events", list(str(e) for e in sim.events))
print(sim.predict_fuel_exhaustion())
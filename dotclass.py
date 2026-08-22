import numpy as np
import sys

class Command:
    NAV, SCAN, CAPTURE, FIRE, LOAD, JETTISON, DETONATE = range(7)
    names = ['NAV', 'SCAN', 'CAPTURE', 'FIRE', 'LOAD', 'JETTISON', 'DETONATE']
    span = range(7)
    def __init__(self, cmd, actor, parameters):
        self. cmd = cmd
        self.actor = actor
        self.parameters = parameters

class Event:
    NAV, SCAN, CAPTURE, FIRE, LOAD, UNLOAD, JETTISON, NO_FUEL, DETONATE, SPAWN = range(10)
    names = ['NAV', 'SCAN', 'CAPTURE', 'FIRE', 'LOAD', 'UNLOAD', 'JETTISON', 'NO_FUEL', 'DETONATE', 'SPAWN']
    def __init__(self, time, evt, entity, parameters):
        self.time = time
        self.evt = evt
        self.entity = entity
        self.parameters = parameters

    def __str__(self):
        return f"T={self.time} {self.entity.name} {self.names[self.evt]} {self.parameters}"

class Capability:
    ENGINE, TANK, BAY, REFINE, DETONATE = range(5)
    span = range(5)
    names = ['ENGINE', 'TANK', 'BAY', 'REFINE', 'DETONATE']

cap_orders = {Capability.ENGINE: [Command.NAV, Command.SCAN, Command.CAPTURE],
              Capability.TANK:   [Command.LOAD, Command.JETTISON],
              Capability.BAY:    [Command.FIRE]}

def mag(n: np.array) -> float:
    return np.linalg.norm(n)

def motion(v: np.array, a: np.array, t: float) -> tuple:
    dr = [v[0] + 0.5*a[0]*t**2,
          v[1] + 0.5*a[1]*t**2,
          v[2] + 0.5*a[2]*t**2]
    dv = [a[0]*t, a[1]*t, a[2]*t]
    return np.array(dr), np.array(dv)

class Entity:
    def __init__(self, name, kind, r, v, a, allegiance, capabilities):
        self.name = name
        self.kind = kind
        self.r, self.v, self.a = r, v, a
        self.allegiance = allegiance
        self.capabilities = capabilities
        self.orders = []

        fc = {}
        for k in self.capabilities:
            if k in Capability.names:
                fc[Capability.names.index(k)] = self.capabilities[k]
            elif k in Capability.span:
                fc[k] = self.capabilities[k]
        self.capabilities = fc


class Simulation:
    def __init__(self, time, entities: list[Entity], orders: list[Command]):
        self.time = time
        self.entities = entities
        self.orders = orders
        self.state_eval = {(0, -1, None)}
        self.events = []

        self.state_eval.update(self.predict_fuel_exhaustion())
        print(self.state_eval)

    def predict_fuel_exhaustion(self):
        fuel_users = self.entities_with_capability([Capability.ENGINE, Capability.TANK])
        predictions = set()
        for e in fuel_users:
            if mag(e.a) > 0:
                fuel = e.capabilities[Capability.TANK]["current"]
                time_to_exhaustion = fuel / mag(e.a)
                prediction = (self.time + time_to_exhaustion, Event.NO_FUEL, e)
                predictions.add(prediction)
        return predictions

    def entities_with_capability(self, capabilities):
        es = set()
        for e in self.entities:
            if all(c in e.capabilities for c in capabilities):
                es.add(e)
        return es

    def run(self, interval = 0):
        state_changes = {(self.time, -1, None), (self.time+interval, -1, None)}
        state_changes.update(self.state_eval)
        state_changes = sorted(state_changes, key= lambda s: s[0])
        new_events = []
        orders = []
        for e in self.entities:
            for o in e.orders:
                if self.time <= o.time <= (self.time + interval):
                    orders.append(o)
                    state_changes.add((o.time, o, e))
        state_changes = filter(lambda sc: sc[0] <= self.time+interval, state_changes)
        now = self.time
        last_start = self.time
        for change_time, reason, entity in state_changes:

            now = change_time
            for e in self.entities:
                dr, dv = motion(e.v, e.a, now - last_start)
                e.r = e.r + dr
                e.v = e.v + dv
                fuel_usage = mag(dv)
                e.capabilities[Capability.TANK]["current"] -= fuel_usage
                print(f"t={now} {e.name} used {fuel_usage}fdv of fuel now at {e.capabilities[Capability.TANK]["current"]}fdv")
                print(f"t={now} {e.name} dr={mag(dr)}m dv={mag(dv)}m/s current v={mag(e.v)}m/s")
            if reason == Event.NO_FUEL:
                print(f"t={now} processing prediction: {entity.name} fuel exhaustion")
                if Capability.TANK in entity.capabilities:
                    e_fuel = entity.capabilities[Capability.TANK]["current"]
                    if e_fuel <= 0:
                        print(f" prediction true {entity.name} at {e_fuel}fdv")
                        entity.capabilities[Capability.TANK]["current"] = 0
                        new_events.append(
                            Event(now, Event.NO_FUEL, entity, {})
                        )
                        entity.a = np.array((0,0,0))
                    else:
                        print(f" prediction false {entity.name} fuel={e_fuel}fdv")
            last_start = now
        self.events += new_events
        self.time = now
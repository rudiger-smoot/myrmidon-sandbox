from __future__ import annotations

from abc import ABC, abstractmethod
import math

import numpy as np

FDV_PREDICTION_ABSOLUTE_TOLERANCE = 1e-12 #todo: APPROXIMATION

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

    def time_til_fuel_exhaustion(self):
        fuel = self.capabilities[Capability.TANK]["current"]
        if mag(self.a):
            time_to_exhaustion = fuel / mag(self.a)
        else: time_to_exhaustion = None
        return time_to_exhaustion

class Command:
    types = BURN, SCAN, CAPTURE, FIRE, LOAD, JETTISON, DETONATE = range(7)
    names = ['BURN', 'SCAN', 'CAPTURE', 'FIRE', 'LOAD', 'JETTISON', 'DETONATE']
    span = range(7)
    def __init__(self, cmd, time: float, actor: Entity, parameters):
        self.cmd = cmd
        self.time = time
        self.actor = actor
        self.parameters = parameters

class Event:
    types = BURN, SCAN, CAPTURE, FIRE, LOAD, UNLOAD, JETTISON, NO_FUEL, DETONATE, SPAWN = range(10)
    names = ['BURN', 'SCAN', 'CAPTURE', 'FIRE', 'LOAD', 'UNLOAD', 'JETTISON', 'NO_FUEL', 'DETONATE', 'SPAWN']
    def __init__(self, time, evt, entity, parameters):
        self.time = time
        self.evt = evt
        self.entity = entity
        self.parameters = parameters

    def __str__(self):
        return f"T={self.time} {self.entity.name} {self.names[self.evt]} {self.parameters}"

class Capability:
    types = ENGINE, TANK, BAY, REFINE, DETONATE = range(5)
    span = range(5)
    names = ['ENGINE', 'TANK', 'BAY', 'REFINE', 'DETONATE']

class Predictor(ABC):
    @abstractmethod
    def predictions(self, sim: Simulation, t: float, evt: Event) -> set[Prediction]:
        pass
    @abstractmethod
    def invalidations(self, sim: Simulation, t: float, evt: Event, queue: set[Prediction]) -> set[Prediction]:
        pass

class Prediction:
    #todo: add hashing magic functions so no duplicates in Set
    def __init__(self, time: float, reason: Event | Command):
        self.time = time
        self.reason = reason

cap_orders = {Capability.ENGINE: [Command.BURN, Command.SCAN, Command.CAPTURE],
              Capability.TANK:   [Command.LOAD, Command.JETTISON],
              Capability.BAY:    [Command.FIRE]}

def mag(n: np.array) -> float:
    return np.linalg.norm(n)


class Simulation:
    def __init__(self, time, entities: list[Entity], orders: list[Command]):
        self.time = time
        self.entities = entities
        self.orders = orders
        self.state_eval = {(0, -1, None)}
        self.events = []

        self.state_eval.update(self.predict_fuel_exhaustion())
        print(self.state_eval)

        self.predictors = {} # map from event type to prediction function
        for event_type in Event.types:
            self.predictors[event_type] = set()

    def motion(v: np.array, a: np.array, t: float) -> tuple:
        dr = [v[0] + 0.5 * a[0] * t ** 2,
              v[1] + 0.5 * a[1] * t ** 2,
              v[2] + 0.5 * a[2] * t ** 2]
        dv = [a[0] * t, a[1] * t, a[2] * t]
        return np.array(dr), np.array(dv)

    def register_predictor(self, event_type: int, p: Predictor) -> bool:
        if event_type in Event.types:
            self.predictors[event_type].add(p)
            return True
        return False

    def fire_event(self, evt: Event):
        predictions = set()
        invalidations = set()
        for predictor in self.predictors[evt.evt]:
            predictions.update(predictor.predictions(self, evt.time, evt))
            invalidations.update(predictor.invalidations(self, evt.time, evt, self.state_eval))
        for invalid_prediction in self.state_eval:
            self.state_eval.remove(invalid_prediction)
        self.state_eval.update(predictions)
        return predictions, invalidations

    def predict_fuel_exhaustion(self):
        fuel_users = self.entities_with_capability([Capability.ENGINE, Capability.TANK])
        predictions = set()
        for e in fuel_users:
            if mag(e.a) > 0:
                time_to_exhaustion = e.time_til_fuel_exhaustion()
                prediction = (self.time + time_to_exhaustion, Event.NO_FUEL, e)
                predictions.add(prediction)
        print(f" T={self.time} adding predictions", predictions)
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
        orders = []
        for e in self.entities:
            for o in e.orders:
                o: Command = o
                if self.time <= o.time <= (self.time + interval):
                    orders.append(o)
                    state_changes.add((o.time, o, e))
        state_changes = sorted(state_changes, key= lambda s: s[0])
        new_events = []
        state_changes = list(filter(lambda sc: sc[0] <= self.time+interval, state_changes))
        now = self.time
        last_start = self.time
        interval_start = now
        interval_end = now + interval
        while state_changes:
            change_time, reason, entity = state_changes.pop(0)
            now = change_time
            self.time = now
            exhausted_entities = set()
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
                    if e_fuel <= 0 or math.isclose(e_fuel, 0, abs_tol=FDV_PREDICTION_ABSOLUTE_TOLERANCE):
                        # todo: deal with floating point comparison tolerances
                        print(f" prediction true {entity.name} at {e_fuel}fdv")
                        entity.capabilities[Capability.TANK]["current"] = 0
                        new_events.append(
                            Event(now, Event.NO_FUEL, entity, {})
                        )
                        exhausted_entities.add(entity)
                    else:
                        print(f" prediction false {entity.name} fuel={e_fuel}fdv")
            if isinstance(reason, Command):
                if reason.cmd == Command.BURN:
                    #print("executing burn")
                    reason: Command = reason
                    entity = reason.actor
                    a = reason.parameters["a"]
                    print(f"t={reason.time} burning {a}")
                    entity.a = a #todo: add acceleration limits
                    new_events.append(Event(now, Event.BURN, entity, reason.parameters))
                    e_fuel = entity.capabilities[Capability.TANK]["current"]
                    predictions = set()
                    if e_fuel > 0: predictions = self.predict_fuel_exhaustion()
                    #todo: generalize "stuff changed, add new predictions" logic.
                    invalidated_scs = []
                    for sc in self.state_eval:
                        # invalidate no-fuel predictions that are later than new exhaustion time
                        if sc[0] > (reason.time + entity.time_til_fuel_exhaustion()) and sc[1] == Event.NO_FUEL and sc[2] == entity:
                            invalidated_scs.append(sc)
                    for isc in invalidated_scs:
                        print(f"invalidating {isc}, popping from interval and state queue")
                        self.state_eval.remove(isc)
                        state_changes.remove(isc)
                    self.state_eval.update(predictions)  # OLD FUEL EXHAUSTION PREDICTIONS INVALID!
                    for p in predictions:
                        predicted_time = p[0]
                        if predicted_time < interval_end:
                            # IMPORTANT CAVEAT CREATED by '<' ! for interval x, only events at start <= t < start+x are processed.
                            # add to stack of unprocessed predictions in this interval's responsibility
                            print(f"adding {p} to interval [{interval_start}, {interval_end})")
                            state_changes.append(p)
                        state_changes = sorted(state_changes, key=lambda s: s[0])
            for e in exhausted_entities:
                # Exhaust acceleration after other orders processed so can't burn at same instant as you run out of fuel
                e.a = np.array((0, 0, 0))
            last_start = now
        self.events += new_events
        self.time = now
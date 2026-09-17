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
    types = BURN, SCAN, CAPTURE, FIRE, LOAD, UNLOAD, JETTISON, NO_FUEL, DETONATE, SPAWN, HEARTBEAT = range(11)
    names = ['BURN', 'SCAN', 'CAPTURE', 'FIRE', 'LOAD', 'UNLOAD', 'JETTISON', 'NO_FUEL', 'DETONATE', 'SPAWN', 'HEARTBEAT']
    def __init__(self, time, evt, actor, parameters):
        self.time = time
        self.evt = evt
        self.actor = actor
        self.parameters: dict = parameters

    def __hash__(self):
        paramset = tuple(self.parameters.keys())
        valset = tuple(self.parameters.values())
        return hash((self.time, self.evt, self.actor, hash(paramset), hash(valset)))

    def __str__(self):
        return f"!!evt {self.time} {self.actor} {self.names[self.evt]}!!"

class Capability:
    types = ENGINE, TANK, BAY, REFINE, DETONATE = range(5)
    span = range(5)
    names = ['ENGINE', 'TANK', 'BAY', 'REFINE', 'DETONATE']

class Predictor(ABC):
    @abstractmethod
    def predictions(self, sim: Simulation, t: float, evt: Event) -> set[tuple]:
        pass
    @abstractmethod
    def invalidations(self, sim: Simulation, t: float, evt: Event, queue: set[tuple]) -> set[tuple]:
        pass

cap_orders = {Capability.ENGINE: [Command.BURN, Command.SCAN, Command.CAPTURE],
              Capability.TANK:   [Command.LOAD, Command.JETTISON],
              Capability.BAY:    [Command.FIRE]}

def mag(n: np.array) -> float:
    return np.linalg.norm(n)

class PredictNofuelFromBurn(Predictor):
    def predictions(self, sim: Simulation, t: float, evt: Event) -> set[tuple]:
        predictions = set()
        entity: Entity = evt.actor
        e_fuel = entity.capabilities[Capability.TANK]["current"]
        if e_fuel > 0 and mag(entity.a) > 0: # if entity has fuel and is burning it
            nofuel_time = t + entity.time_til_fuel_exhaustion()
            nofuel_prediction = (nofuel_time, Event(nofuel_time, Event.NO_FUEL, entity, {}))
            predictions.add(nofuel_prediction)
            print(f" predicting from {t}s: {nofuel_prediction}")
        return predictions

    def invalidations(self, sim: Simulation, t: float, evt: Event, queue: set[tuple]) -> set[tuple]:
        invalidations = set()
        for prediction in queue:
            if isinstance(prediction[1], Event):
                # invalidate no-fuel predictions
                if prediction[1].actor == evt.actor and prediction[1].evt == Event.NO_FUEL:
                    invalidations.add(prediction)
        return invalidations


class Simulation:
    predictors = {}  # map from event type to prediction function
    predictors[Event.BURN] = {PredictNofuelFromBurn()}
    for event_type in Event.types:
        if not event_type in predictors.keys():
            predictors[event_type] = set()
    def __init__(self, time, entities: list[Entity], orders: list[Command]):
        self.time = time
        self.entities = entities
        self.orders = orders
        # predictions in state_eval are tuples of (time, reason), with reason an Event or Command
        self.state_eval = {(0, Event(0, Event.HEARTBEAT, None, {}))}
        self.events = []

        self.state_eval.update(self.predict_fuel_exhaustion())
        print(self.state_eval)

    def motion(self, v: np.array, a: np.array, t: float) -> tuple:
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

    def update_predictions(self, evt: Event):
        predictions = set()
        invalidations = set()
        queue = set(filter(lambda sc: sc[0] >= evt.time, self.state_eval))
        for predictor in self.predictors[evt.evt]:
            predictions.update(predictor.predictions(None, evt.time, evt))
            invalidations.update(predictor.invalidations(self, evt.time, evt, queue))
        self.state_eval = self.state_eval.difference(invalidations)
        self.state_eval.update(predictions)
        return predictions, invalidations

    def predict_fuel_exhaustion(self):
        fuel_users = self.entities_with_capability([Capability.ENGINE, Capability.TANK])
        predictions = set()
        for e in fuel_users:
            if mag(e.a) > 0:
                time_to_exhaustion = e.time_til_fuel_exhaustion()
                absolute_time = self.time +time_to_exhaustion
                prediction = (absolute_time, Event(absolute_time, Event.NO_FUEL, e, {}))
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
        state_changes = {(self.time, Event(self.time, Event.HEARTBEAT, None, {})),
        (self.time+interval, Event(self.time+interval, Event.HEARTBEAT, None, {}))}
        state_changes.update(self.state_eval)
        orders = []
        for e in self.entities:
            for o in e.orders:
                o: Command = o
                if self.time <= o.time <= (self.time + interval):
                    orders.append(o)
                    state_changes.add((o.time, o))
        state_changes = sorted(state_changes, key= lambda s: s[0])
        new_events = []
        state_changes = list(filter(lambda sc: sc[0] <= self.time+interval, state_changes))
        now = self.time
        last_start = self.time
        interval_start = now
        interval_end = now + interval
        while state_changes:
            change_time, reason = state_changes.pop(0)
            actor = reason.actor
            now = change_time
            self.time = now
            exhausted_entities = set()
            new_predictions, invalidations = set(), set()
            for e in self.entities:
                dr, dv = self.motion(e.v, e.a, now - last_start)
                e.r = e.r + dr
                e.v = e.v + dv
                fuel_usage = mag(dv)
                e.capabilities[Capability.TANK]["current"] -= fuel_usage
                print(f"t={now} {e.name} used {fuel_usage}fdv of fuel now at {e.capabilities[Capability.TANK]["current"]}fdv")
                print(f"t={now} {e.name} a={mag(e.a)} dr={mag(dr)}m dv={mag(dv)}m/s current v={mag(e.v)}m/s")
            if isinstance(reason, Event):
                if reason.evt == Event.NO_FUEL:
                    if not actor.name:
                        pass
                    print(f"t={now} processing prediction: {actor.name} fuel exhaustion")
                    if Capability.TANK in actor.capabilities:
                        e_fuel = actor.capabilities[Capability.TANK]["current"]
                        if math.isclose(e_fuel, 0, abs_tol=FDV_PREDICTION_ABSOLUTE_TOLERANCE):
                            # todo: deal with floating point comparison tolerances
                            print(f" prediction true {actor.name} at {e_fuel}fdv")
                            actor.capabilities[Capability.TANK]["current"] = 0
                            new_events.append(
                                Event(now, Event.NO_FUEL, actor, {})
                            )
                            actor.a = np.array((0, 0, 0))
                        else:
                            print(f" prediction false {actor.name} fuel={e_fuel}fdv")
            elif isinstance(reason, Command):
                if reason.cmd == Command.BURN:
                    #print("executing burn")
                    reason: Command = reason
                    a = reason.parameters["a"]
                    print(f"t={reason.time} burning {a}")
                    e_fuel = actor.capabilities[Capability.TANK]["current"]
                    if e_fuel:
                        actor.a = a #todo: add acceleration limits
                    else:
                        actor.a = np.array((0, 0, 0))
                    new_events.append(Event(now, Event.BURN, actor, reason.parameters))
                    new_predictions, invalidations = self.update_predictions(Event(now, Event.BURN, actor, reason.parameters))
            # update state changes queue for this interval based on processed prediction's consequences
            for isc in invalidations:
                state_changes.remove(isc)
            for p in new_predictions:
                predicted_time = p[0]
                if predicted_time < interval_end:
                    # IMPORTANT CAVEAT CREATED by '<' ! for interval x, only events at start <= t < start+x are processed.
                    # add to stack of unprocessed predictions in this interval's responsibility
                    print(f"adding {p} to interval [{interval_start}, {interval_end})")
                    state_changes.append(p)
                state_changes = sorted(state_changes, key=lambda s: s[0])
            last_start = now
        self.events += new_events
        self.time = now
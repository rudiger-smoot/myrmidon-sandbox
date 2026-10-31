from __future__ import annotations

from abc import ABC, abstractmethod
import math
from wsgiref.util import request_uri

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
        self.event_log = []
        self.last_child_id = 0

        fc = {}
        for k in self.capabilities:
            if k in Capability.names:
                fc[Capability.names.index(k)] = self.capabilities[k]
            elif k in Capability.span:
                fc[k] = self.capabilities[k]
        self.capabilities = fc

    def get_capability(self, cap_type):
        return self.capabilities.get(cap_type, None)

    def time_til_fuel_exhaustion(self):
        fuel = self.capabilities[Capability.TANK]["current"]
        if mag(self.a):
            time_to_exhaustion = fuel / mag(self.a)
        else: time_to_exhaustion = None
        return time_to_exhaustion

    def witness_event(self, evt: Event):
        self.event_log.append(evt)

    def spawn_child(self, kind, acc, caps) -> Entity:
        child = Entity(f"{self.name}.{self.last_child_id}", kind,
                         self.r, self.v, acc, self.allegiance,caps)
        self.last_child_id += 1
        return child

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
    types = BURN, SCAN, CAPTURE, FIRE, LOAD, UNLOAD, JETTISON, NO_FUEL, DETONATE, SPAWN, HEARTBEAT, GONE = range(12)
    names = ['BURN', 'SCAN', 'CAPTURE', 'FIRE', 'LOAD', 'UNLOAD', 'JETTISON', 'NO_FUEL', 'DETONATE', 'SPAWN', 'HEARTBEAT',
             'GONE']
    def __init__(self, time, evt, actor, parameters):
        self.time = time
        self.evt = evt
        self.actor = actor
        self.parameters: dict = parameters

        self.ok = True
        self.result = {}

    def __hash__(self):
        paramset = tuple(self.parameters.keys())
        valset = tuple(self.parameters.values())
        return hash((self.time, self.evt, self.actor, hash(paramset), hash(valset)))

    def __repr__(self):
        return f"!!evt {self.time} {self.actor.name if self.actor else '?'} {self.names[self.evt]} {self.parameters}!!"

class Capability:
    types = ENGINE, TANK, BAY, REFINE, SUPPLY, DETONATE, SCANNER, DEEP_SPACE_SCANNER = range(8)
    span = range(8)
    names = ['ENGINE', 'TANK', 'BAY', 'REFINE', 'SUPPLY', 'DETONATE', 'SCANNER', 'DEEP_SPACE_SCANNER']

class Predictor(ABC):
    @abstractmethod
    def predictions(self, sim: Simulation, t: float, evt: Event) -> set[tuple]:
        pass
    @abstractmethod
    def invalidations(self, sim: Simulation, t: float, evt: Event, queue: set[tuple]) -> set[tuple]:
        pass

cap_orders = {Capability.ENGINE:  [Command.BURN, Command.CAPTURE],
              Capability.TANK:    [Command.LOAD, Command.JETTISON],
              Capability.BAY:     [Command.FIRE],
              Capability.SCANNER: [Command.SCAN]}

def mag(n: np.array) -> float:
    return np.linalg.norm(n)

class PredictNofuel(Predictor):
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

class PredictNothingFromDestruction(Predictor):
    def predictions(self, sim: Simulation, t: float, evt: Event) -> set[tuple]:
        return set()

    def invalidations(self, sim: Simulation, t: float, evt: Event, queue: set[tuple]) -> set[tuple]:
        invalidations = set()
        for prediction in queue:
            if prediction[1].actor == evt.actor:
                    invalidations.add(prediction)
        return invalidations

class Rules:
    INTERACTION_RANGE_LIMIT = 1000 # meters
    INTERACTION_SPEED_LIMIT = 100 # m/s
    SCAN_RANGE = 150000 # meters
    DEEP_SCAN_RANGE = 150000*2 # meters
    MAX_SHIP_IMPULSE = 1235 # m/s/s, equal to dotwar1 limit
    MAX_MISSILE_IMPULSE = MAX_SHIP_IMPULSE * 2 # m/s/s
    LIGHTSPEED = 1079251200 * 1/60 * 1000# m/s, equal to dotwar1 limit
    SPEED_LIMIT = LIGHTSPEED * 0.9 # .9c

class Simulation:
    predictors = {}  # map from event type to prediction function
    predictors[Event.BURN] = {PredictNofuel()}
    predictors[Event.LOAD] = {PredictNofuel()}
    predictors[Event.GONE] = {PredictNothingFromDestruction()}
    for event_type in Event.types:
        if not event_type in predictors.keys():
            predictors[event_type] = set()
    rules = Rules
    def __init__(self, time, entities: list[Entity], orders: list[Command]):
        self.time = time
        self.entities = entities
        self.orders = orders
        # predictions in state_eval are tuples of (time, reason), with reason an Event or Command
        self.state_eval = {(0, Event(0, Event.HEARTBEAT, None, {}))}
        self.events = []

        self.state_eval.update(self.predict_fuel_exhaustion())
        print(self.state_eval)

    def motion(self, v: np.array, a: np.array, dt: float) -> tuple:
        dr = [(v[0] * dt) + 0.5 * a[0] * (dt ** 2),
              (v[1] * dt) + 0.5 * a[1] * (dt ** 2),
              (v[2] * dt) + 0.5 * a[2] * (dt ** 2)]
        dv = [a[0] * dt, a[1] * dt, a[2] * dt]
        return np.array(dr), np.array(dv)

    def register_predictor(self, event_type: int, p: Predictor) -> bool:
        if event_type in Event.types:
            self.predictors[event_type].add(p)
            return True
        return False

    def realize_event(self, evt: Event):
        self.events.append(evt)
        return self.updated_predictions(evt)

    def updated_predictions(self, evt: Event):
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

    def entity_by_name(self, name: str) -> Entity:
        e = None
        for e in self.entities:
            if e.name == name:
                return e
        return e

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
            destroyed_entities = set()
            new_predictions, invalidations = set(), set()
            for e in self.entities:
                dr, dv = self.motion(e.v, e.a, now - last_start)
                e.r = e.r + dr
                e.v = e.v + dv
                if e.get_capability(Capability.TANK):
                    fuel_usage = mag(dv)
                    e.capabilities[Capability.TANK]["current"] -= fuel_usage
            if isinstance(reason, Event):
                if reason.evt == Event.NO_FUEL:
                    print(f"t={now} processing prediction: {actor.name} fuel exhaustion")
                    if Capability.TANK in actor.capabilities:
                        e_fuel = actor.capabilities[Capability.TANK]["current"]
                        if math.isclose(e_fuel, 0, abs_tol=FDV_PREDICTION_ABSOLUTE_TOLERANCE) or e_fuel < 0:
                            print(f" prediction true {actor.name} at {e_fuel}fdv")
                            if e_fuel < 0: print(" warning: fuel negative")
                            if abs(e_fuel) <= 1: print(" warning: floating point comparison")
                            actor.capabilities[Capability.TANK]["current"] = 0
                            new_events.append(
                                Event(now, Event.NO_FUEL, actor, {})
                            )
                            actor.a = np.array((0, 0, 0))
                        else:
                            print(f" prediction false {actor.name} fuel={e_fuel}fdv")
            elif isinstance(reason, Command):
                reason: Command = reason
                if reason.cmd == Command.SCAN:
                    range = 0
                    if Capability.SCANNER in reason.actor.capabilities:
                        range = self.rules.SCAN_RANGE
                    elif Capability.DEEP_SPACE_SCANNER in reason.actor.capabilities:
                        range = self.rules.DEEP_SCAN_RANGE
                    event_out = Event(reason.time, Event.SCAN, reason.actor, reason.parameters)
                    event_out.ok = True
                    event_out.result = {'range': range, 'contacts':[]}
                    for e in self.entities:
                        contact = {}
                        if e is not reason.actor and mag(e.r - actor.r) <= range:
                            print(f"{actor.name} just saw {e.name} @ {mag(e.r - actor.r)}m away")
                            contact['name'] = e.name
                            contact['kinematics'] = {'r': e.r, 'v': e.v, 'a': e.a}
                            contact['kind'] = e.kind
                            contact['allegiance'] = e.allegiance
                            event_out.result['contacts'].append(contact)
                    if reason.actor.get_capability(Capability.DEEP_SPACE_SCANNER):
                        for e in self.entities:
                            if e is not reason.actor and e.allegiance == reason.actor.allegiance:
                                e.witness_event(event_out)
                    new_events.append(event_out)
                elif reason.cmd == Command.BURN:
                    a = reason.parameters["a"]
                    print(f"t={reason.time} burning {a}")
                    e_fuel = actor.capabilities[Capability.TANK]["current"]
                    if e_fuel:
                        if mag(a) > self.rules.MAX_SHIP_IMPULSE:
                            actor.a = self.rules.MAX_SHIP_IMPULSE * (a/mag(a))
                            print(f"limited burn of {mag(a)}m/s to {self.rules.MAX_SHIP_IMPULSE}m/s")
                        else: actor.a = a
                    else:
                        actor.a = np.array((0, 0, 0))
                    new_events.append(Event(now, Event.BURN, actor, reason.parameters))
                    updates = self.updated_predictions(Event(now, Event.BURN, actor, reason.parameters))
                    new_predictions.update(updates[0])
                    invalidations.update(updates[1])
                elif reason.cmd == Command.LOAD:
                    print("executing load")
                    event_out = Event(reason.time, Event.LOAD, reason.actor, reason.parameters)
                    target_candidates = tuple(filter(lambda e: mag(actor.r - e.r) <= self.rules.SCAN_RANGE, self.entities))
                    target = None
                    for tc in target_candidates:
                        if tc.name == reason.parameters["target"]:
                            target = tc
                    if target is None:
                        event_out.ok = False
                        event_out.result = {'msg':f'no visible entity named {reason.parameters["target"]}'}
                    else:
                        distance = mag(actor.r - target.r)
                        vel_diff = mag(actor.v - target.v)
                        print(actor.r, actor.v, distance)
                        ok = True
                        if distance > self.rules.INTERACTION_RANGE_LIMIT:
                            event_out.ok = ok = False
                            event_out.result = {'msg': f'target is greater than {self.rules.INTERACTION_RANGE_LIMIT}m away'}
                        if vel_diff > self.rules.INTERACTION_SPEED_LIMIT:
                            event_out.ok = ok = False
                            event_out.result = {'msg': f'you are {vel_diff}m/s faster than target, must be within {self.rules.INTERACTION_SPEED_LIMIT}m/s'}
                        if not (Capability.TANK in target.capabilities and
                                (Capability.REFINE in target.capabilities or Capability.SUPPLY in target.capabilities)):
                            event_out.ok = ok = False
                            event_out.result = {
                                'msg': f'target must be planet or fuel cache'}
                        if Capability.REFINE in target.capabilities and target.allegiance != actor.allegiance:
                            event_out.ok = ok = False
                            event_out.result = {
                                'msg': f"can't reload from non-allied planet"}
                        if ok:
                            event_out.result = {'fuel_added': 0, 'missiles_added': 0}
                            # load fuel and missiles
                            e_tank = actor.capabilities[Capability.TANK]
                            t_caps = target.capabilities
                            if e_tank["current"] < e_tank["max"]:
                                missing_fuel = e_tank["max"] - e_tank["current"]
                                withdraw_fuel = min(missing_fuel, t_caps[Capability.TANK]["current"])
                                target.capabilities[Capability.TANK]["current"] -= withdraw_fuel
                                e_tank["current"] += withdraw_fuel
                                print(f"refueled {actor.name} from {target.name} for {withdraw_fuel}fdv")
                                event_out.ok = True
                                event_out.result['fuel_added'] = withdraw_fuel
                                if Capability.SUPPLY in t_caps and target.capabilities[Capability.TANK]["current"] == 0:
                                    # fuel cache exhausted
                                    print("used up all of fuel cache")
                                    death_event = Event(reason.time, Event.GONE, target, {"cause": event_out})
                                    updates = self.updated_predictions(death_event)
                                    new_predictions.update(updates[0])
                                    invalidations.update(updates[1])
                                    new_events.append(death_event)
                                    destroyed_entities.add(target)
                            if Capability.BAY in actor.capabilities and Capability.REFINE in target.capabilities:
                                e_bay = actor.capabilities[Capability.BAY]
                                if e_bay["current"] < e_bay["max"]:
                                    missing = e_bay["max"] - e_bay["current"]
                                    e_bay["current"] = e_bay["max"]
                                    event_out.ok = True
                                    event_out.result['missiles_added'] = missing
                    updates = self.updated_predictions(event_out)
                    new_predictions.update(updates[0])
                    invalidations.update(updates[1])
                    new_events.append(event_out)
                    print(event_out.ok, event_out.result)
                elif reason.cmd == Command.FIRE:
                    print("processing fire event")
                    event_out = Event(reason.time, Event.FIRE, reason.actor, reason.parameters)
                    actor = reason.actor
                    if not actor.get_capability(Capability.BAY) or actor.get_capability(Capability.BAY)["current"] == 0:
                        event_out.ok = False
                        event_out.result['msg'] = "you have no torpedo bay or 0 torpedoes"
                    else:
                        # spawn torp and deplete supply
                        ta, countdown = reason.parameters['a'], reason.parameters['countdown']
                        if mag(ta) > self.rules.MAX_MISSILE_IMPULSE:
                            ta = self.rules.MAX_MISSILE_IMPULSE * (ta / mag(ta))
                            print(f"limited torpedo burn of {mag(ta)}m/s/s to {self.rules.MAX_MISSILE_IMPULSE}m/s/s")
                        torpedo = actor.spawn_child("torpedo", ta, {Capability.DETONATE:{'countdown':countdown}})
                        self.entities.append(torpedo)
                        print(f"{actor.name} fired torpedo {torpedo.name}")
                    updates = self.updated_predictions(event_out)
                    new_predictions.update(updates[0])
                    invalidations.update(updates[1])
                    new_events.append(event_out)

            # update state changes queue for this interval based on processed prediction's consequences
            for entity in destroyed_entities:
                self.entities.remove(entity)
            for isc in invalidations:
                try:
                    state_changes.remove(isc)
                except:
                    pass
            for p in new_predictions:
                predicted_time = p[0]
                if predicted_time < interval_end:
                    # IMPORTANT CAVEAT IMPLIED by '<' : for interval x, only events at start <= t < start+x are processed.
                    # add to stack of unprocessed predictions in this interval's responsibility
                    print(f"adding {p} to interval [{interval_start}, {interval_end})")
                    state_changes.append(p)
                state_changes = sorted(state_changes, key=lambda s: s[0])
            last_start = now
        for evt in new_events:
            for e in self.entities:
                if (mag(evt.actor.r - e.r) <= self.rules.SCAN_RANGE) and (evt not in e.event_log):
                    # todo: don't witness failed orders
                    e.witness_event(evt)
        self.events += new_events
        self.time = now
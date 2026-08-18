from __future__ import annotations

import unittest

import numpy as np

from src.model.domain import Delivery, ProblemData, Route, VehicleType
from src.model.evaluator import RouteEvaluator, evaluate_solution
from src.model.policy_q2 import Q2Policy
from src.solver.fleet import select_vehicles
from src.solver.greedy import build_greedy_routes, validate_solution
from src.solver.local_search import (
    eliminate_low_load_routes,
    improve_routes_merge,
    improve_routes_relocate,
    repack_split_low_load_route,
)
from src.solver.savings import build_savings_routes
from src.solver.scheduling import (
    assign_physical_vehicles,
    select_and_schedule_multitrip,
    validate_vehicle_schedule,
)


def _problem() -> ProblemData:
    return ProblemData(
        distance=np.array(
            [
                [0.0, 10.0, 12.0],
                [10.0, 0.0, 3.0],
                [12.0, 3.0, 0.0],
            ]
        ),
        demands={1: (4000.0, 8.0), 2: (1000.0, 2.0)},
        windows={1: (480.0, 900.0), 2: (480.0, 900.0)},
        coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
        all_customer_ids=(1, 2),
    )


class BaselineTests(unittest.TestCase):
    def test_q2_policy_flags_fuel_arrival_in_green_zone(self) -> None:
        problem = ProblemData(
            distance=np.array([[0.0, 10.0], [10.0, 0.0]]),
            demands={1: (100.0, 1.0)},
            windows={1: (480.0, 900.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0)},
            all_customer_ids=(1,),
            green_customer_ids=frozenset({1}),
        )
        fuel = VehicleType("FUEL", "fuel", 1000.0, 5.0, 1)
        route = Route(fuel, 1, [Delivery(1, 100.0, 1.0)])
        result = evaluate_solution(
            [route],
            RouteEvaluator(problem, policy=Q2Policy(frozenset({1}))),
            optimize_departures=False,
        )
        self.assertEqual(result.policy_violation_count, 1)
        self.assertEqual(result.routes[0].stops[0].policy_violation_reason,
                         "fuel_vehicle_in_green_zone_during_restricted_period")

    def test_q2_policy_allows_electric_arrival_in_green_zone(self) -> None:
        problem = _problem()
        problem = ProblemData(
            distance=problem.distance,
            demands=problem.demands,
            windows=problem.windows,
            coordinates=problem.coordinates,
            all_customer_ids=problem.all_customer_ids,
            green_customer_ids=frozenset({1, 2}),
        )
        electric = VehicleType("EV", "electric", 3000.0, 10.0, 1)
        route = Route(electric, 1, [Delivery(1, 100.0, 1.0)])
        result = evaluate_solution(
            [route],
            RouteEvaluator(problem, policy=Q2Policy(frozenset({1, 2}))),
            optimize_departures=False,
        )
        self.assertEqual(result.policy_violation_count, 0)

    def test_q2_policy_half_open_boundary_times(self) -> None:
        policy = Q2Policy(frozenset({1}))
        cases = (
            (479.0, False),
            (480.0, True),
            (959.0, True),
            (960.0, False),
        )
        for arrival_minutes, expected in cases:
            with self.subTest(arrival_minutes=arrival_minutes):
                self.assertEqual(
                    policy.violates("fuel", 1, arrival_minutes),
                    expected,
                )

    def test_split_delivery_and_capacity(self) -> None:
        problem = _problem()
        fleet = (VehicleType("TEST", "electric", 3000.0, 10.0, 2),)
        routes = build_greedy_routes(problem, fleet)
        self.assertEqual(len(routes), 2)
        self.assertTrue(all(route.total_weight <= 3000.0 + 1e-8 for route in routes))
        delivered = sum(item.weight for route in routes for item in route.deliveries)
        self.assertAlmostEqual(delivered, 5000.0, places=6)

    def test_evaluator_returns_complete_cost(self) -> None:
        problem = _problem()
        fleet = (VehicleType("TEST", "electric", 3000.0, 10.0, 2),)
        routes = build_greedy_routes(problem, fleet)
        result = RouteEvaluator(problem).best_departure(routes[0])
        self.assertGreater(result.distance_km, 0)
        self.assertGreater(result.energy_cost, 0)
        self.assertGreater(result.carbon_cost, 0)
        self.assertGreaterEqual(result.total_cost, result.fixed_cost)

    def test_solution_cost_components_add_up(self) -> None:
        problem = _problem()
        fleet = (VehicleType("TEST", "electric", 3000.0, 10.0, 2),)
        routes = build_greedy_routes(problem, fleet)
        result = evaluate_solution(routes, RouteEvaluator(problem))
        component_total = (
            result.fixed_cost
            + result.energy_cost
            + result.carbon_cost
            + result.waiting_cost
            + result.late_cost
        )
        self.assertAlmostEqual(result.total_cost, component_total, places=8)
        self.assertEqual(result.vehicle_count, len(routes))

    def test_energy_is_per_100km_and_uses_current_load(self) -> None:
        problem = _problem()
        evaluator = RouteEvaluator(problem)
        _, empty_energy = evaluator.travel_leg(10.0, 600.0, "electric", 0.0)
        _, full_energy = evaluator.travel_leg(10.0, 600.0, "electric", 1.0)
        expected_empty = 10.0 / 100.0 * evaluator._energy_per_100km("electric", 35.4)
        self.assertAlmostEqual(empty_energy, expected_empty, places=8)
        self.assertAlmostEqual(full_energy / empty_energy, 1.35, places=8)

    def test_inter_route_relocate_can_remove_a_vehicle(self) -> None:
        problem = ProblemData(
            distance=np.array(
                [
                    [0.0, 10.0, 11.0],
                    [10.0, 0.0, 1.0],
                    [11.0, 1.0, 0.0],
                ]
            ),
            demands={1: (40.0, 1.0), 2: (40.0, 1.0)},
            windows={1: (480.0, 900.0), 2: (480.0, 900.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
            all_customer_ids=(1, 2),
        )
        vehicle = VehicleType("TEST", "electric", 100.0, 5.0, 2)
        routes = [
            Route(vehicle, 1, [Delivery(1, 40.0, 1.0)]),
            Route(vehicle, 2, [Delivery(2, 40.0, 1.0)]),
        ]
        improved = improve_routes_relocate(routes, RouteEvaluator(problem))
        self.assertEqual(len(improved), 1)
        self.assertAlmostEqual(improved[0].total_weight, 80.0)

    def test_low_load_elimination_and_merge_remove_vehicle(self) -> None:
        problem = ProblemData(
            distance=np.array(
                [
                    [0.0, 10.0, 11.0],
                    [10.0, 0.0, 1.0],
                    [11.0, 1.0, 0.0],
                ]
            ),
            demands={1: (40.0, 1.0), 2: (40.0, 1.0)},
            windows={1: (480.0, 900.0), 2: (480.0, 900.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
            all_customer_ids=(1, 2),
        )
        vehicle = VehicleType("TEST", "electric", 100.0, 5.0, 2)
        routes = [
            Route(vehicle, 1, [Delivery(1, 40.0, 1.0)]),
            Route(vehicle, 2, [Delivery(2, 40.0, 1.0)]),
        ]
        evaluator = RouteEvaluator(problem)
        eliminated = eliminate_low_load_routes(routes, evaluator)
        self.assertEqual(len(eliminated), 1)

        routes = [
            Route(vehicle, 1, [Delivery(1, 40.0, 1.0)]),
            Route(vehicle, 2, [Delivery(2, 40.0, 1.0)]),
        ]
        merged = improve_routes_merge(routes, evaluator)
        self.assertEqual(len(merged), 1)

    def test_global_vehicle_selection_honors_counts(self) -> None:
        problem = ProblemData(
            distance=np.array([[0.0, 5.0], [5.0, 0.0]]),
            demands={1: (80.0, 1.0)},
            windows={1: (480.0, 900.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0)},
            all_customer_ids=(1,),
        )
        electric = VehicleType("EV", "electric", 100.0, 5.0, 1)
        fuel = VehicleType("FUEL", "fuel", 100.0, 5.0, 1)
        placeholder = VehicleType("TEMP", "fuel", 100.0, 5.0, 2)
        routes = [
            Route(placeholder, 1, [Delivery(1, 40.0, 0.5)]),
            Route(placeholder, 2, [Delivery(1, 40.0, 0.5)]),
        ]
        selected = select_vehicles(
            routes, RouteEvaluator(problem), (electric, fuel)
        )
        self.assertEqual({route.vehicle_type.name for route in selected}, {"EV", "FUEL"})

    def test_savings_builder_preserves_total_demand(self) -> None:
        problem = _problem()
        fleet = (VehicleType("TEST", "electric", 3000.0, 10.0, 2),)
        routes = build_savings_routes(problem, fleet)
        self.assertEqual(len(routes), 2)
        self.assertAlmostEqual(
            sum(item.weight for route in routes for item in route.deliveries),
            5000.0,
            places=6,
        )

    def test_one_physical_vehicle_can_execute_two_nonoverlapping_trips(self) -> None:
        problem = ProblemData(
            distance=np.array(
                [
                    [0.0, 5.0, 5.0],
                    [5.0, 0.0, 2.0],
                    [5.0, 2.0, 0.0],
                ]
            ),
            demands={1: (40.0, 1.0), 2: (40.0, 1.0)},
            windows={1: (480.0, 540.0), 2: (900.0, 960.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
            all_customer_ids=(1, 2),
        )
        vehicle = VehicleType("TEST", "electric", 100.0, 5.0, 1)
        routes = [
            Route(vehicle, 1, [Delivery(1, 40.0, 1.0)]),
            Route(vehicle, 2, [Delivery(2, 40.0, 1.0)]),
        ]
        evaluator = RouteEvaluator(problem)
        assign_physical_vehicles(routes, evaluator)
        result = evaluate_solution(routes, evaluator)
        self.assertEqual(result.trip_count, 2)
        self.assertEqual(result.vehicle_count, 1)
        self.assertAlmostEqual(result.fixed_cost, 400.0)
        self.assertEqual({route.vehicle_number for route in routes}, {1})
        self.assertEqual({route.trip_number for route in routes}, {1, 2})
        validate_vehicle_schedule(routes, evaluator)

    def test_multitrip_selection_does_not_cap_daily_trip_count(self) -> None:
        problem = ProblemData(
            distance=np.array(
                [
                    [0.0, 2.0, 2.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                ]
            ),
            demands={1: (80.0, 1.0), 2: (80.0, 1.0)},
            windows={1: (480.0, 900.0), 2: (480.0, 900.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
            all_customer_ids=(1, 2),
        )
        vehicle = VehicleType("EV", "electric", 100.0, 5.0, 1)
        placeholder = VehicleType("TEMP", "fuel", 100.0, 5.0, 99)
        routes = [
            Route(placeholder, 0, [Delivery(1, 80.0, 1.0)]),
            Route(placeholder, 0, [Delivery(2, 80.0, 1.0)]),
        ]
        evaluator = RouteEvaluator(problem)
        scheduled = select_and_schedule_multitrip(
            routes,
            evaluator,
            (vehicle,),
            max_physical_vehicles=1,
        )
        result = evaluate_solution(scheduled, evaluator, optimize_departures=False)
        self.assertEqual(result.trip_count, 2)
        self.assertEqual(result.vehicle_count, 1)
        self.assertAlmostEqual(result.fixed_cost, 400.0)
        validate_vehicle_schedule(scheduled, evaluator)

    def test_savings_uses_max_trip_capacity_not_fleet_trip_slots(self) -> None:
        problem = ProblemData(
            distance=np.array([[0.0, 5.0], [5.0, 0.0]]),
            demands={1: (6000.0, 6.0)},
            windows={1: (480.0, 1200.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0)},
            all_customer_ids=(1,),
        )
        vehicle = VehicleType("EV", "electric", 3000.0, 15.0, 1)
        routes = build_savings_routes(problem, (vehicle,))
        self.assertEqual(len(routes), 2)
        self.assertTrue(all(route.total_weight == 3000.0 for route in routes))

    def test_multitrip_schedule_rejects_cross_day_delivery(self) -> None:
        problem = ProblemData(
            distance=np.array([[0.0, 100.0], [100.0, 0.0]]),
            demands={1: (80.0, 1.0)},
            windows={1: (480.0, 1200.0)},
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0)},
            all_customer_ids=(1,),
        )
        vehicle = VehicleType("EV", "electric", 100.0, 5.0, 1)
        route = Route(vehicle, 1, [Delivery(1, 80.0, 1.0)])
        evaluator = RouteEvaluator(problem)
        with self.assertRaises(ValueError):
            evaluator.best_departure(
                route,
                earliest=23.0 * 60.0,
                latest_finish=24.0 * 60.0,
            )

    def test_split_repack_can_remove_low_load_trip(self) -> None:
        problem = ProblemData(
            distance=np.array(
                [
                    [0.0, 5.0, 5.0, 5.0],
                    [5.0, 0.0, 1.0, 1.0],
                    [5.0, 1.0, 0.0, 1.0],
                    [5.0, 1.0, 1.0, 0.0],
                ]
            ),
            demands={1: (40.0, 0.4), 2: (80.0, 0.8), 3: (80.0, 0.8)},
            windows={
                1: (480.0, 1200.0),
                2: (480.0, 1200.0),
                3: (480.0, 1200.0),
            },
            coordinates={
                0: (0.0, 0.0),
                1: (1.0, 0.0),
                2: (2.0, 0.0),
                3: (3.0, 0.0),
            },
            all_customer_ids=(1, 2, 3),
        )
        vehicle = VehicleType("EV", "electric", 100.0, 5.0, 3)
        routes = [
            Route(vehicle, 1, [Delivery(1, 40.0, 0.4)]),
            Route(vehicle, 2, [Delivery(2, 80.0, 0.8)]),
            Route(vehicle, 3, [Delivery(3, 80.0, 0.8)]),
        ]
        repacked = repack_split_low_load_route(routes, RouteEvaluator(problem))
        self.assertEqual(len(repacked), 2)
        delivered = sum(item.weight for route in repacked for item in route.deliveries)
        self.assertAlmostEqual(delivered, 200.0)
        self.assertTrue(all(route.total_weight <= 100.0 + 1e-7 for route in repacked))

    def test_multitrip_can_enforce_exact_electric_trip_count(self) -> None:
        problem = ProblemData(
            distance=np.array(
                [
                    [0.0, 2.0, 2.0, 2.0],
                    [2.0, 0.0, 1.0, 1.0],
                    [2.0, 1.0, 0.0, 1.0],
                    [2.0, 1.0, 1.0, 0.0],
                ]
            ),
            demands={1: (40.0, 1.0), 2: (40.0, 1.0), 3: (40.0, 1.0)},
            windows={
                1: (480.0, 1200.0),
                2: (480.0, 1200.0),
                3: (480.0, 1200.0),
            },
            coordinates={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0), 3: (3.0, 0.0)},
            all_customer_ids=(1, 2, 3),
        )
        electric = VehicleType("EV", "electric", 100.0, 5.0, 1)
        fuel = VehicleType("FUEL", "fuel", 100.0, 5.0, 3)
        placeholder = VehicleType("TEMP", "fuel", 100.0, 5.0, 99)
        routes = [
            Route(placeholder, 0, [Delivery(customer, 40.0, 1.0)])
            for customer in (1, 2, 3)
        ]
        scheduled = select_and_schedule_multitrip(
            routes,
            RouteEvaluator(problem),
            (electric, fuel),
            electric_trip_minimum=2,
            electric_trip_limit=2,
        )
        self.assertEqual(
            sum(route.vehicle_type.propulsion == "electric" for route in scheduled),
            2,
        )


if __name__ == "__main__":
    unittest.main()

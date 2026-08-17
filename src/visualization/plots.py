from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

from src.model.domain import ProblemData, Route
from src.model.evaluator import SolutionEvaluation


FONT = "Microsoft YaHei, SimHei, Arial, sans-serif"
COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "teal": "#72B7B2",
    "grid": "#D9DEE7",
    "text": "#263238",
}


def _svg(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<title>{escape(title)}</title>',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<g font-family="{FONT}" fill="{COLORS["text"]}">',
            *body,
            "</g>",
            "</svg>",
        ]
    )


def _write(path: Path, width: int, height: int, title: str, body: list[str]) -> Path:
    path.write_text(_svg(width, height, title, body), encoding="utf-8")
    return path


def plot_cost_breakdown(solution: SolutionEvaluation, output_dir: Path, prefix: str) -> Path:
    labels = ["启动", "能源", "碳", "等待", "迟到"]
    values = [solution.fixed_cost, solution.energy_cost, solution.carbon_cost, solution.waiting_cost, solution.late_cost]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["teal"], COLORS["red"]]
    width, height = 900, 500
    left, top, plot_width = 120, 90, 600
    maximum = max(values) or 1.0
    body = [f'<text x="{width / 2}" y="38" text-anchor="middle" font-size="22" font-weight="500">问题一成本构成（总成本 {solution.total_cost:,.2f} 元）</text>']
    for index, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        y = top + index * 72
        bar_width = value / maximum * plot_width
        share = value / solution.total_cost * 100.0
        body.extend([
            f'<text x="{left - 15}" y="{y + 25}" text-anchor="end" font-size="15">{label}</text>',
            f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="34" fill="{color}"/>',
            f'<text x="{left + bar_width + 10:.2f}" y="{y + 24}" font-size="14">{value:,.0f} 元（{share:.1f}%）</text>',
        ])
    return _write(output_dir / f"{prefix}_cost_breakdown.svg", width, height, "问题一成本构成", body)


def plot_routes(routes: list[Route], problem: ProblemData, output_dir: Path, prefix: str) -> Path:
    width, height, pad = 920, 720, 70
    coordinates = list(problem.coordinates.values())
    min_x, max_x = min(x for x, _ in coordinates), max(x for x, _ in coordinates)
    min_y, max_y = min(y for _, y in coordinates), max(y for _, y in coordinates)

    def point(node: int) -> tuple[float, float]:
        x, y = problem.coordinates[node]
        sx = pad + (x - min_x) / max(max_x - min_x, 1e-9) * (width - 2 * pad)
        sy = height - pad - (y - min_y) / max(max_y - min_y, 1e-9) * (height - 2 * pad)
        return sx, sy

    body = [f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="22" font-weight="500">问题一优化配送路线空间分布</text>']
    for route in routes:
        nodes = [0, *(item.customer_id for item in route.deliveries), 0]
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(node) for node in nodes))
        color = COLORS["green"] if route.vehicle_type.propulsion == "electric" else COLORS["orange"]
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1" opacity="0.25"/>')
    max_demand = max(weight for weight, _ in problem.demands.values())
    for node in problem.active_customer_ids:
        x, y = point(node)
        radius = 3.0 + 5.0 * problem.demands[node][0] / max_demand
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{COLORS["blue"]}" stroke="white" stroke-width="0.7"/>')
    depot_x, depot_y = point(0)
    body.extend([
        f'<circle cx="{depot_x:.2f}" cy="{depot_y:.2f}" r="10" fill="{COLORS["red"]}"/>',
        f'<text x="{depot_x + 13:.2f}" y="{depot_y + 5:.2f}" font-size="14">配送中心</text>',
        f'<line x1="70" y1="680" x2="110" y2="680" stroke="{COLORS["green"]}" stroke-width="4"/><text x="118" y="685" font-size="13">新能源车</text>',
        f'<line x1="220" y1="680" x2="260" y2="680" stroke="{COLORS["orange"]}" stroke-width="4"/><text x="268" y="685" font-size="13">燃油车</text>',
    ])
    return _write(output_dir / f"{prefix}_route_map.svg", width, height, "配送路线空间分布", body)


def plot_vehicle_gantt(routes: list[Route], solution: SolutionEvaluation, output_dir: Path, prefix: str) -> Path:
    jobs: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for route, result in zip(routes, solution.routes, strict=True):
        vehicle_id = f"{route.vehicle_type.name}-{route.vehicle_number:03d}"
        jobs[vehicle_id].append((result.start_minutes, result.finish_minutes, route.vehicle_type.propulsion))
    vehicle_ids = sorted(jobs, key=lambda name: ("EV" not in name, name))
    width, row_height = 1100, 24
    height = max(520, 100 + row_height * len(vehicle_ids))
    left, right, top = 160, 35, 65
    min_time = 480.0
    max_time = max(finish for vehicle_jobs in jobs.values() for _, finish, _ in vehicle_jobs)

    def x_position(minutes: float) -> float:
        return left + (minutes - min_time) / max(max_time - min_time, 1.0) * (width - left - right)

    body = [f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="22" font-weight="500">物理车辆多趟配送甘特图</text>']
    tick = 480
    while tick <= max_time + 1e-9:
        x = x_position(tick)
        label = f"D+{int(tick // 1440)} {int(tick % 1440 // 60):02d}:00" if tick >= 1440 else f"{int(tick // 60):02d}:00"
        body.extend([
            f'<line x1="{x:.2f}" y1="{top - 8}" x2="{x:.2f}" y2="{height - 35}" stroke="{COLORS["grid"]}" stroke-width="1"/>',
            f'<text x="{x:.2f}" y="{height - 12}" text-anchor="middle" font-size="11">{label}</text>',
        ])
        tick += 120
    for index, vehicle_id in enumerate(vehicle_ids):
        y = top + index * row_height
        body.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" font-size="11">{vehicle_id}</text>')
        for start, finish, propulsion in sorted(jobs[vehicle_id]):
            x = x_position(start)
            bar_width = max(1.0, x_position(finish) - x)
            color = COLORS["green"] if propulsion == "electric" else COLORS["orange"]
            body.append(f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="17" fill="{color}"/>')
    return _write(output_dir / f"{prefix}_vehicle_gantt.svg", width, height, "物理车辆多趟配送甘特图", body)


def plot_load_rates(routes: list[Route], output_dir: Path, prefix: str) -> Path:
    width, height, left, top, size = 720, 660, 90, 65, 520
    body = [
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="22" font-weight="500">各配送趟次重量与体积装载率</text>',
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="{COLORS["grid"]}"/>',
    ]
    threshold_x = left + 0.8 * size
    threshold_y = top + 0.2 * size
    body.extend([
        f'<line x1="{threshold_x}" y1="{top}" x2="{threshold_x}" y2="{top + size}" stroke="{COLORS["red"]}" stroke-dasharray="6 5"/>',
        f'<line x1="{left}" y1="{threshold_y}" x2="{left + size}" y2="{threshold_y}" stroke="{COLORS["red"]}" stroke-dasharray="6 5"/>',
    ])
    for route in routes:
        weight_rate = route.total_weight / route.vehicle_type.capacity_weight
        volume_rate = route.total_volume / route.vehicle_type.capacity_volume
        x = left + min(weight_rate, 1.0) * size
        y = top + (1.0 - min(volume_rate, 1.0)) * size
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{COLORS["blue"]}" opacity="0.7"/>')
    for value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = left + value * size
        y = top + (1.0 - value) * size
        body.extend([
            f'<text x="{x:.2f}" y="{top + size + 23}" text-anchor="middle" font-size="12">{value:.1f}</text>',
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{value:.1f}</text>',
        ])
    body.extend([
        f'<text x="{left + size / 2}" y="{height - 18}" text-anchor="middle" font-size="15">重量装载率</text>',
        f'<text x="24" y="{top + size / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 24 {top + size / 2})">体积装载率</text>',
    ])
    return _write(output_dir / f"{prefix}_load_rates.svg", width, height, "配送趟次装载率", body)


def plot_optimization_trace(trace: list[dict[str, float | int | str]], output_dir: Path, prefix: str) -> Path:
    width, height, left, top = 980, 520, 90, 70
    plot_width, plot_height = 830, 340
    stages = [str(row["stage"]) for row in trace]
    costs = [float(row["total_cost"]) for row in trace]
    minimum, maximum = min(costs), max(costs)
    span = max(maximum - minimum, 1.0)
    points: list[str] = []
    body = [f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="22" font-weight="500">优化过程成本变化</text>']
    for index, (stage, cost) in enumerate(zip(stages, costs, strict=True)):
        x = left + index / max(len(costs) - 1, 1) * plot_width
        y = top + (maximum - cost) / span * plot_height
        points.append(f"{x:.2f},{y:.2f}")
        body.extend([
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{COLORS["blue"]}"/>',
            f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="end" font-size="10" transform="rotate(-28 {x:.2f} {top + plot_height + 25})">{escape(stage)}</text>',
        ])
    joined_points = " ".join(points)
    body.insert(1, f'<polyline points="{joined_points}" fill="none" stroke="{COLORS["blue"]}" stroke-width="2"/>')
    body.append(f'<text x="25" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 25 {top + plot_height / 2})">总成本（元）</text>')
    return _write(output_dir / f"{prefix}_optimization_trace.svg", width, height, "优化过程成本变化", body)


def plot_solution_figures(
    routes: list[Route],
    solution: SolutionEvaluation,
    problem: ProblemData,
    trace: list[dict[str, float | int | str]],
    output_root: Path,
    prefix: str = "question1_optimized",
) -> list[Path]:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_cost_breakdown(solution, figure_dir, prefix),
        plot_routes(routes, problem, figure_dir, prefix),
        plot_vehicle_gantt(routes, solution, figure_dir, prefix),
        plot_load_rates(routes, figure_dir, prefix),
        plot_optimization_trace(trace, figure_dir, prefix),
    ]

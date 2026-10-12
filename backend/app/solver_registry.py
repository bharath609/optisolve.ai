from collections.abc import Callable
from typing import Any

from .optimizer import solve_model, solve_transportation
from .schemas import BusinessModel


Solver = Callable[[Any], dict[str, Any]]


def _solve_production(model: Any) -> dict[str, Any]:
    products, resources, total_profit = solve_model(model)
    return {"products": products, "resources": resources, "total_profit": total_profit}


def _solve_transportation(model: Any) -> dict[str, Any]:
    shipments, total_cost = solve_transportation(model)
    return {"shipments": shipments, "total_cost": total_cost}


def _solve_procurement(model: Any) -> dict[str, Any]:
    from .optimizer import solve_procurement

    purchases, total_cost, discount_applied = solve_procurement(model)
    return {"purchases": purchases, "total_cost": total_cost, "discount_applied": discount_applied}


def _solve_game(model: Any) -> dict[str, Any]:
    from .optimizer import solve_game

    strategies, game_value = solve_game(model)
    return {"strategies": strategies, "game_value": game_value}


SOLVERS: dict[str, Solver] = {
    "production": _solve_production,
    "transportation": _solve_transportation,
    "procurement": _solve_procurement,
    "game_theory": _solve_game,
}


def solve_business_model(business_model: BusinessModel) -> dict[str, Any]:
    solver = SOLVERS.get(business_model.model_type)
    if solver is None:
        supported = ", ".join(sorted(SOLVERS))
        raise ValueError(f"No solver is registered for {business_model.model_type}. Supported models: {supported}.")
    return solver(business_model.model)
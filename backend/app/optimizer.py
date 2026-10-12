import pyomo.environ as pyo

from .schemas import GameModel, PlanningModel, ProcurementModel, ProductResult, PurchaseResult, ResourceResult, ShipmentResult, StrategyResult, TransportationModel


def solve_model(planning_model: PlanningModel) -> tuple[list[ProductResult], list[ResourceResult], float]:
    model = pyo.ConcreteModel()
    product_names = [product.name for product in planning_model.products]
    resource_names = list(planning_model.capacities)
    products = {product.name: product for product in planning_model.products}

    model.products = pyo.Set(initialize=product_names)
    model.quantity = pyo.Var(model.products, domain=pyo.NonNegativeReals)

    def resource_constraint(model: pyo.ConcreteModel, resource: str):
        return sum(products[name].resource_usage.get(resource, 0) * model.quantity[name] for name in model.products) <= planning_model.capacities[resource]

    model.resources = pyo.Set(initialize=resource_names)
    model.capacity = pyo.Constraint(model.resources, rule=resource_constraint)

    def demand_constraint(model: pyo.ConcreteModel, name: str):
        limit = products[name].demand_limit
        return pyo.Constraint.Skip if limit is None else model.quantity[name] <= limit

    model.demand = pyo.Constraint(model.products, rule=demand_constraint)
    model.profit = pyo.Objective(
        expr=sum(products[name].profit * model.quantity[name] for name in model.products),
        sense=pyo.maximize,
    )

    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError("The HiGHS solver is not available in this environment.")
    result = solver.solve(model)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not find an optimal plan: {result.solver.termination_condition}")

    product_results = [
        ProductResult(name=name, quantity=round(pyo.value(model.quantity[name]), 4), profit=round(products[name].profit * pyo.value(model.quantity[name]), 2))
        for name in product_names
    ]
    resource_results = []
    for resource in resource_names:
        used = sum(products[name].resource_usage.get(resource, 0) * pyo.value(model.quantity[name]) for name in product_names)
        capacity = planning_model.capacities[resource]
        resource_results.append(ResourceResult(name=resource, used=round(used, 4), capacity=capacity, utilization=round(used / capacity if capacity else 0, 4)))

    return product_results, resource_results, round(pyo.value(model.profit), 2)


def solve_transportation(transport_model: TransportationModel) -> tuple[list[ShipmentResult], float]:
    model = pyo.ConcreteModel()
    source_names = [source.name for source in transport_model.sources]
    destination_names = [destination.name for destination in transport_model.destinations]
    supplies = {source.name: source.supply for source in transport_model.sources}
    demands = {destination.name: destination.demand for destination in transport_model.destinations}

    model.sources = pyo.Set(initialize=source_names)
    model.destinations = pyo.Set(initialize=destination_names)
    model.quantity = pyo.Var(model.sources, model.destinations, domain=pyo.NonNegativeReals)
    model.supply = pyo.Constraint(model.sources, rule=lambda current, source: sum(current.quantity[source, destination] for destination in current.destinations) <= supplies[source])
    model.demand = pyo.Constraint(model.destinations, rule=lambda current, destination: sum(current.quantity[source, destination] for source in current.sources) >= demands[destination])
    model.cost = pyo.Objective(
        expr=sum(transport_model.costs[source][destination] * model.quantity[source, destination] for source in source_names for destination in destination_names),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError("The HiGHS solver is not available in this environment.")
    result = solver.solve(model)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not find an optimal transportation plan: {result.solver.termination_condition}")

    shipments = [
        ShipmentResult(
            source=source,
            destination=destination,
            quantity=round(pyo.value(model.quantity[source, destination]), 4),
            cost=round(transport_model.costs[source][destination] * pyo.value(model.quantity[source, destination]), 2),
        )
        for source in source_names
        for destination in destination_names
        if pyo.value(model.quantity[source, destination]) > 0.0001
    ]
    return shipments, round(pyo.value(model.cost), 2)


def solve_game(game_model: GameModel) -> tuple[list[StrategyResult], float]:
    model = pyo.ConcreteModel()
    player_one_actions = game_model.player_one_actions
    player_two_actions = game_model.player_two_actions
    model.player_one_actions = pyo.Set(initialize=player_one_actions)
    model.player_two_actions = pyo.Set(initialize=player_two_actions)
    model.probability = pyo.Var(model.player_one_actions, domain=pyo.NonNegativeReals)
    model.value = pyo.Var()
    model.probability_sum = pyo.Constraint(expr=sum(model.probability[action] for action in model.player_one_actions) == 1)
    model.response = pyo.Constraint(
        model.player_two_actions,
        rule=lambda current, response: sum(game_model.payoffs[action][response] * current.probability[action] for action in current.player_one_actions) >= current.value,
    )
    model.objective = pyo.Objective(expr=model.value, sense=pyo.maximize)

    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError("The HiGHS solver is not available in this environment.")
    result = solver.solve(model)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not find an optimal game strategy: {result.solver.termination_condition}")

    strategies = [
        StrategyResult(player="Player 1", action=action, probability=round(pyo.value(model.probability[action]), 4))
        for action in player_one_actions
        if pyo.value(model.probability[action]) > 0.0001
    ]
    return strategies, round(pyo.value(model.value), 4)


def _procurement_lp(proc_model: ProcurementModel, extra_spend: tuple[str, float] | None = None) -> tuple[dict[tuple[str, str], float], float]:
    """Minimum base cost (and quantities) to meet all requirements, optionally
    forcing spend at one shop above its discount threshold."""
    model = pyo.ConcreteModel()
    pairs = [(shop, item) for shop in proc_model.shops for item in proc_model.items]
    model.pairs = pyo.Set(initialize=pairs, dimen=2)
    model.quantity = pyo.Var(model.pairs, domain=pyo.NonNegativeReals)
    prices = proc_model.prices

    model.demand = pyo.Constraint(proc_model.items, rule=lambda current, item: sum(current.quantity[shop, item] for shop in proc_model.shops) >= proc_model.requirements[item])
    for shop in proc_model.shops:
        for item in proc_model.items:
            if item not in prices.get(shop, {}):
                model.quantity[shop, item].fix(0)
    if extra_spend is not None:
        spend_shop, threshold = extra_spend
        model.threshold = pyo.Constraint(
            expr=sum(prices[spend_shop].get(item, 0) * model.quantity[spend_shop, item] for item in proc_model.items) >= threshold
        )
    model.cost = pyo.Objective(
        expr=sum(prices[shop].get(item, 0) * model.quantity[shop, item] for shop, item in pairs),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError("The HiGHS solver is not available in this environment.")
    result = solver.solve(model)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not find an optimal purchase plan: {result.solver.termination_condition}")
    quantities = {(shop, item): round(max(0, pyo.value(model.quantity[shop, item])), 4) for shop, item in pairs}
    return quantities, round(pyo.value(model.cost), 2)


def solve_procurement(proc_model: ProcurementModel) -> tuple[list[PurchaseResult], float, float]:
    """Best purchase plan, comparing the plain cheapest plan against each
    shop-threshold-discount scenario. Returns (purchases, total_cost, discount_applied)."""
    quantities, base_cost = _procurement_lp(proc_model)
    best_quantities, best_total, best_discount = quantities, base_cost, 0.0
    for rule in proc_model.discounts:
        if rule.shop not in proc_model.shops:
            continue
        try:
            alt_quantities, alt_base = _procurement_lp(proc_model, (rule.shop, rule.threshold))
        except RuntimeError:
            continue
        alt_total = round(alt_base - rule.amount, 2)
        if alt_total < best_total - 1e-9:
            best_quantities, best_total, best_discount = alt_quantities, alt_total, rule.amount
    purchases = [
        PurchaseResult(
            shop=shop,
            item=item,
            quantity=quantity,
            cost=round(proc_model.prices[shop][item] * quantity, 2),
        )
        for (shop, item), quantity in best_quantities.items()
        if quantity > 0.0001 and item in proc_model.prices.get(shop, {})
    ]
    return purchases, best_total, best_discount

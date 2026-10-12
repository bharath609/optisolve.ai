from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    profit: float
    resource_usage: dict[str, float]
    demand_limit: float | None = Field(default=None, ge=0)

    @field_validator("profit")
    @classmethod
    def profit_must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("profit must be non-negative")
        return value


class PlanningModel(BaseModel):
    products: list[Product] = Field(min_length=1, max_length=30)
    capacities: dict[str, float]

    @field_validator("capacities")
    @classmethod
    def capacities_must_be_positive(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(capacity < 0 for capacity in value.values()):
            raise ValueError("capacities must contain non-negative values")
        return value


class TransportSource(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    supply: float = Field(ge=0)


class TransportDestination(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    demand: float = Field(ge=0)


class TransportationModel(BaseModel):
    sources: list[TransportSource] = Field(min_length=1, max_length=50)
    destinations: list[TransportDestination] = Field(min_length=1, max_length=50)
    costs: dict[str, dict[str, float]]

    @field_validator("costs")
    @classmethod
    def costs_must_be_non_negative(cls, value: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        if any(cost < 0 for destination_costs in value.values() for cost in destination_costs.values()):
            raise ValueError("transportation costs must be non-negative")
        return value


class GameModel(BaseModel):
    player_one_actions: list[str] = Field(min_length=2, max_length=30)
    player_two_actions: list[str] = Field(min_length=2, max_length=30)
    payoffs: dict[str, dict[str, float]]
    zero_sum: bool = True


class DiscountRule(BaseModel):
    shop: str = Field(min_length=1, max_length=80)
    amount: float = Field(ge=0)
    threshold: float = Field(ge=0)


class ProcurementModel(BaseModel):
    shops: list[str] = Field(min_length=1, max_length=30)
    items: list[str] = Field(min_length=1, max_length=50)
    requirements: dict[str, float]
    prices: dict[str, dict[str, float]]
    discounts: list[DiscountRule] = Field(default_factory=list)

    @field_validator("requirements", "prices")
    @classmethod
    def values_must_be_non_negative(cls, value: dict) -> dict:
        def check(node: object) -> None:
            if isinstance(node, dict):
                for child in node.values():
                    check(child)
            elif isinstance(node, (int, float)) and node < 0:
                raise ValueError("procurement quantities and prices must be non-negative")
        check(value)
        return value


class BusinessModel(BaseModel):
    model_type: Literal["production", "transportation", "procurement", "game_theory"]
    model: PlanningModel | TransportationModel | ProcurementModel | GameModel
    problem_summary: str = ""
    technique: str = ""
    objective: str = ""
    variables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ProblemRequest(BaseModel):
    problem: str = Field(min_length=10, max_length=10000)


class ProductResult(BaseModel):
    name: str
    quantity: float
    profit: float


class ResourceResult(BaseModel):
    name: str
    used: float
    capacity: float
    utilization: float


class ShipmentResult(BaseModel):
    source: str
    destination: str
    quantity: float
    cost: float


class PurchaseResult(BaseModel):
    shop: str
    item: str
    quantity: float
    cost: float


class StrategyResult(BaseModel):
    player: str
    action: str
    probability: float


class AnalyzeResponse(BaseModel):
    model_type: Literal["production", "transportation", "procurement", "game_theory"]
    model: PlanningModel | TransportationModel | ProcurementModel | GameModel
    problem_summary: str = ""
    technique: str = ""
    objective: str = ""
    variables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    products: list[ProductResult] = Field(default_factory=list)
    resources: list[ResourceResult] = Field(default_factory=list)
    shipments: list["ShipmentResult"] = Field(default_factory=list)
    purchases: list[PurchaseResult] = Field(default_factory=list)
    strategies: list[StrategyResult] = Field(default_factory=list)
    game_value: float | None = None
    total_profit: float | None = None
    total_cost: float | None = None
    discount_applied: float = 0
    currency: str = "$"
    explanation: str

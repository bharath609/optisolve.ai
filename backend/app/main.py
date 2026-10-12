import os

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .ai import detect_currency, extract_model
from .solver_registry import solve_business_model
from .schemas import AnalyzeResponse, ProblemRequest

load_dotenv()

app = FastAPI()

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        os.getenv(
            "FRONTEND_ORIGIN",
            "http://localhost:5173,https://optiforge-ai.vercel.app,https://optiforge.ai,https://www.optiforge.ai,https://optisolve.vercel.app,https://optisolve.ai,https://www.optisolve.ai",
        ),
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins + ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "OptiSolve backend is working"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: ProblemRequest):
    currency = detect_currency(request.problem)
    try:
        business_model = await extract_model(request.problem)
        solution = solve_business_model(business_model)
        if business_model.model_type == "procurement":
            discount = solution["discount_applied"]
            per_item = []
            for item in business_model.model.items:
                got = [p for p in solution["purchases"] if p.item == item]
                per_item.append(" + ".join(f"{p.quantity:g} {item} from {p.shop}" for p in got))
            explanation = f"Buy {'; '.join(per_item)}."
            if discount > 0:
                explanation += f" A shop discount of {currency}{discount:,.2f} applies. Cheapest possible total: {currency}{solution['total_cost']:,.2f}."
            else:
                explanation += f" Cheapest possible total: {currency}{solution['total_cost']:,.2f}."
            return AnalyzeResponse(
                model_type=business_model.model_type,
                model=business_model.model,
                problem_summary=business_model.problem_summary,
                technique=business_model.technique,
                objective=business_model.objective,
                variables=business_model.variables,
                constraints=business_model.constraints,
                assumptions=business_model.assumptions,
                purchases=solution["purchases"],
                total_cost=solution["total_cost"],
                discount_applied=discount,
                currency=currency,
                explanation=explanation,
            )
        if business_model.model_type == "transportation":
            return AnalyzeResponse(
                model_type=business_model.model_type,
                model=business_model.model,
                problem_summary=business_model.problem_summary,
                technique=business_model.technique,
                objective=business_model.objective,
                variables=business_model.variables,
                constraints=business_model.constraints,
                assumptions=business_model.assumptions,
                shipments=solution["shipments"],
                total_cost=solution["total_cost"],
                currency=currency,
                explanation=f"The lowest-cost transportation plan has a total shipping cost of {currency}{solution['total_cost']:,.2f}.",
            )
        if business_model.model_type == "game_theory":
            return AnalyzeResponse(
                model_type=business_model.model_type,
                model=business_model.model,
                problem_summary=business_model.problem_summary,
                technique=business_model.technique,
                objective=business_model.objective,
                variables=business_model.variables,
                constraints=business_model.constraints,
                assumptions=business_model.assumptions,
                strategies=solution["strategies"],
                game_value=solution["game_value"],
                currency=currency,
                explanation=f"The optimal mixed strategy guarantees an expected payoff of {solution['game_value']:.4g} for Player 1.",
            )
        products = solution["products"]
        resources = solution["resources"]
        total_profit = solution["total_profit"]
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    constrained_resources = [resource.name for resource in resources if resource.utilization >= 0.999]
    if constrained_resources:
        explanation = f"The plan produces the highest possible profit of {currency}{total_profit:,.2f}. " \
            f"The limiting resource is {', '.join(constrained_resources)}."
    else:
        explanation = f"The plan produces the highest possible profit of {currency}{total_profit:,.2f} within the stated limits."
    return AnalyzeResponse(
        model_type=business_model.model_type,
        model=business_model.model,
        problem_summary=business_model.problem_summary,
        technique=business_model.technique,
        objective=business_model.objective,
        variables=business_model.variables,
        constraints=business_model.constraints,
        assumptions=business_model.assumptions,
        products=products,
        resources=resources,
        total_profit=total_profit,
        currency=currency,
        explanation=explanation,
    )
import json
import os
import re
import asyncio
from urllib.request import Request, urlopen

from .schemas import BusinessModel


def detect_currency(text: str) -> str:
    if "₹" in text:
        return "₹"
    if re.search(r"\bRs\.?", text, re.I):
        return "₹"
    if "€" in text:
        return "€"
    if "£" in text:
        return "£"
    if "$" in text:
        return "$"
    return "$"


async def extract_model(problem: str) -> BusinessModel:
    if not _is_complex_problem(problem):
        simple_key = os.getenv("SIMPLE_AI_API_KEY")
        if simple_key:
            try:
                return await _extract_with_provider(
                    problem,
                    simple_key,
                    os.getenv("SIMPLE_AI_BASE_URL", "https://api.groq.com/openai/v1"),
                    os.getenv("SIMPLE_AI_MODEL", "llama-3.1-8b-instant"),
                )
            except Exception:
                pass
        try:
            return _extract_with_fallback(problem)
        except ValueError as error:
            raise ValueError(str(error)) from error

    api_key = os.getenv("EXPLABS_API_KEY")
    if api_key:
        try:
            return await _extract_with_provider(problem, api_key)
        except Exception:
            return _extract_with_fallback(problem)
    return _extract_with_fallback(problem)


def _is_complex_problem(problem: str) -> bool:
    text = problem.strip()
    numeric_values = re.findall(r"-?\d+(?:\.\d+)?", text)
    return len(text) > 700 or len(text.split()) > 120 or len(numeric_values) > 16


FREE_FORM_GUIDANCE = """Users phrase problems freely in their own words — never require rigid templates like "Product A".
They may write casually ("I run a bakery...", "I ship from...", "my factory has...").
Understand synonyms: profit = profit/margin/earn/yield/contribution/gain/revenue/makes/gives/price;
needs = needs/uses/requires/consumes/takes/spends; have = have/available/in stock/on hand/capacity/total/limited to/left.
Product names can be anything (chairs, bread, phones, X1...). Resource names can be anything
(wood, machine hours, labour, oven...). Places can be anything (Warehouse A, Mumbai depot, Store X).
Costs can appear as "A to X costs 4", "A->X 4", "$4 from A to X", "cost from A to X is 4".
Normalize them, keep numbers exact, never invent numbers.
First trace the problem: production (make things for profit with limited resources),
transportation (ship from supplies to demands with route costs),
procurement (buy minimum quantities cheapest across shops with unit prices and possible threshold discounts),
or game theory (two players, payoffs).
If the text is general math (equations, calculus) rather than production/transportation/game-theory
optimization, return {"error":"unsupported"}."""


async def _extract_with_provider(
    problem: str,
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
) -> BusinessModel:
    base_url = (base_url or os.getenv("EXPERIENTIAL_BASE_URL", "https://api.experientiallabs.ai/v1")).rstrip("/")
    model = model or os.getenv("EXPERIENTIAL_MODEL", "gpt-6-astra")
    prompt = """You are an operations-research model classifier and mathematical formulation assistant. Read the business problem and return JSON only.

Choose model_type as production, transportation, procurement, or game_theory. Return this envelope:
{"model_type":"production|transportation|procurement|game_theory","problem_summary":string,"technique":string,"objective":string,"variables":[string],"constraints":[string],"assumptions":[string],"model":{...}}

For production, model must be {"products":[{"name":string,"profit":number,"resource_usage":{resource:number},"demand_limit":number|null}],"capacities":{resource:number}}.
For transportation, model must be {"sources":[{"name":string,"supply":number}],"destinations":[{"name":string,"demand":number}],"costs":{"source":{"destination":number}}}.
For procurement (groceries, diet, vendor choice: meet minimum needs cheapest across shops), model must be {"shops":[string],"items":[string],"requirements":{item:number},"prices":{"shop":{item:number}},"discounts":[{"shop":string,"amount":number,"threshold":number}]}. List items in requirement order and map "respectively" prices to that order.
For game_theory, model must be {"player_one_actions":[string],"player_two_actions":[string],"payoffs":{"player_one_action":{"player_two_action":number}},"zero_sum":true}. Use Player 1's payoff values and preserve negative payoffs.
Use consistent names, preserve all stated numbers, and do not invent missing numbers. If the problem is neither supported type, return a concise JSON error with {"error":"unsupported"}.

""" + FREE_FORM_GUIDANCE + "\n\nUser text:\n" + problem
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    def call_provider() -> str:
        with urlopen(request, timeout=45) as response:
            payload = json.loads(response.read())
        return payload["choices"][0]["message"]["content"]

    content = await asyncio.to_thread(call_provider)
    payload = json.loads(_strip_code_fence(content))
    if payload.get("error") == "unsupported":
        raise ValueError("OptiSolve only solves optimization problems — production planning, transportation, shopping/purchase planning, and game theory — not general math. Describe profits with limited resources, shipping supplies/demands/costs, shopping needs with shop prices, or a game payoff, in any wording.")
    return BusinessModel.model_validate(payload)


def _strip_code_fence(content: str) -> str:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    return match.group(0) if match else content


STOP_WORDS = {
    "the", "a", "an", "each", "every", "per", "of", "for", "with", "and", "or",
    "to", "from", "in", "on", "is", "are", "it", "its", "we", "our", "you",
    "that", "this", "which", "using", "uses", "use", "needs", "need", "requires",
    "require", "consumes", "consume", "takes", "take", "profit", "profits",
    "margin", "earn", "earns", "gives", "give", "yields", "makes", "make",
    "units", "unit", "hours", "hour", "hrs", "hr", "kg", "kgs", "tons", "ton",
}


def _clean_name(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw.strip())
    s = re.sub(r"^(the|a|an|each|every|our|your|my|and|both|from|to|at|in|of)\s+", "", s, flags=re.I)
    # strip trailing verbs/adjectives captured by lazy regexes ("Factory A can" -> "Factory A")
    s = re.sub(
        r"\s+(can|has|have|with|holds?|stocks?|suppl\w*|supply|produces?|makes?|offers?|provides?|needs?|requires?|demands?|demand|orders?|wants?)$",
        "", s, flags=re.I,
    )
    return s.strip()


def _canon(s: str) -> str:
    norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
    stripped = re.sub(r"\s+", " ", re.sub(
        r"\b(units?|hours?|hrs?|hr|kg|kgs|tons?|tonnes?|days?|minutes?|mins?)\b", " ", norm,
    )).strip()
    return stripped or norm


def _profit_from_sentence(sentence: str) -> float | None:
    m = re.search(
        r"(?:profit|margin|earning|contribution|gain)\s+(?:per|for|of|on)\s+[A-Za-z0-9\- ]+?\s+(?:is|=|:)\s*\$?\s*(\d+(?:\.\d+)?)",
        sentence, re.I,
    )
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:profit|margin|earned|earnings|contribution)", sentence, re.I)
    if m:
        return float(m.group(1))
    m = re.search(
        r"(?:profit|margin|earning|contribution|gain|revenue|nets?|earns?|yields?|makes?|gives?|price|value)"
        r"(?:\s*(?:per|for|of|on|is|:|=))?\s*\$?\s*(\d+(?:\.\d+)?)",
        sentence, re.I,
    )
    if m:
        return float(m.group(1))
    return None


def _product_name_from_sentence(sentence: str, fallback: str) -> str:
    m = re.search(r"(product\s+[A-Za-z0-9\-]+)", sentence, re.I)
    if m:
        return _clean_name(m.group(1))
    # "Profit per X1 is 40" — exact single-token name (avoids lazy-match traps like "X")
    m = re.search(
        r"(?:profit|margin|earning|contribution|gain)\s+(?:per|for|of|on)\s+(?:of\s+)?([A-Za-z][A-Za-z0-9\-]*)",
        sentence, re.I,
    )
    if m and m.group(1).strip().lower() not in STOP_WORDS:
        return _clean_name(m.group(1))
    # "Mango Smoothie for a profit of 50" — up to 3 words before "for a profit"
    m = re.search(
        r"([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,2})\s+for\s+a\s+(?:profit|margin)\s+of\s+\$?\d",
        sentence, re.I,
    )
    if m:
        cand = _clean_name(re.sub(r"^(sells?|makes?|produces?|offers?|manufactures?|provides?)\s+", "", m.group(1), flags=re.I))
        last = (cand.lower().split(" ") or [""])[-1]
        if cand and len(cand) <= 30 and last not in STOP_WORDS:
            return cand
    m = re.search(
        r"([A-Za-z][A-Za-z0-9\- ]{0,28}?)\s+(?:gives?|earns?|yields?|makes?|brings?|fetches?|nets?)\s+(?:a\s+)?(?:profit\s+(?:of\s+)?)?\$?\d",
        sentence, re.I,
    )
    if m and m.group(1).strip().lower() not in STOP_WORDS:
        cand = _clean_name(m.group(1))
        cand = re.sub(r"^(we|they|you|i)\s+", "", cand, flags=re.I).strip()
        cand = re.sub(r"^(make|makes|produces?|manufactures?|sells?|offers?|builds?)\s+", "", cand, flags=re.I).strip()
        cand = re.sub(r"\s+(that|which|who)$", "", cand, flags=re.I).strip()
        if cand and cand.lower() not in STOP_WORDS:
            return cand
    m = re.search(
        r"(?:profit|margin|earning|contribution|gain)(?:\s*(?:per|for|of|on))?\s+(?:of\s+)?([A-Za-z][A-Za-z0-9\- ]{0,28}?)\s+(?:is|=|:|of)\s*\$?\d",
        sentence, re.I,
    )
    if m and m.group(1).strip().lower() not in STOP_WORDS:
        cand = _clean_name(re.sub(r"\s+(is|are|of|at)$", "", m.group(1), flags=re.I))
        if cand and len(cand) <= 30:
            return cand
    m = re.search(r"([A-Za-z][A-Za-z0-9\- ]{0,28}?)\s*[:\-–]\s*\$?\d+(?:\.\d+)?\s*(?:profit|margin)", sentence, re.I)
    if m:
        return _clean_name(m.group(1))
    # "bread margin 25" / "chairs profit 40" — name before the keyword
    m = re.search(
        r"([A-Za-z][A-Za-z0-9\-]*)\s+(?:profit|margin|earning|contribution)s?\s*(?:is|=|:|of)?\s*\$?\d+(?:\.\d+)?",
        sentence, re.I,
    )
    if m and m.group(1).strip().lower() not in STOP_WORDS:
        return _clean_name(m.group(1))
    m = re.search(r"(?:each|every|one|per)\s+([A-Za-z][A-Za-z0-9\-]*)", sentence, re.I)
    if m and m.group(1).strip().lower() not in STOP_WORDS:
        return _clean_name(m.group(1))
    return fallback


def _usage_free_form(segment: str, exclude_value: float | None) -> dict[str, float]:
    usage: dict[str, float] = {}
    skipped = False
    for m in re.finditer(
        r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|kg|kgs|tons?|tonnes?|units?|days?|minutes?|mins?)?\s*(?:of\s+)?([A-Za-z][A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|\.|$|\+|needs?|uses?|requires?|consumes?|takes?|per))",
        segment, re.I,
    ):
        value = float(m.group(1))
        resource = re.sub(r"\s+per\s+unit.*$", "", m.group(3) or "", flags=re.I).strip()
        resource = re.sub(r"^(units?|hours?|hrs?|kg)\s+", "", resource, flags=re.I).strip()
        low = resource.lower()
        if not resource or len(resource) > 40:
            continue
        if re.match(r"^(per|each|every)\b", resource, re.I):
            continue
        if "profit" in low or "margin" in low or "earn" in low or "revenue" in low:
            continue
        if low in STOP_WORDS:
            continue
        if exclude_value is not None and not skipped and abs(value - exclude_value) < 1e-9:
            skipped = True
            continue
        name = _clean_name(resource)
        if name:
            usage[name] = value
    return usage


def _parse_capacities_free(text: str) -> dict[str, float]:
    caps: dict[str, float] = {}
    sentences = text.replace("$", "").split(".")
    for raw in re.split(r"[.\n;]+", text.replace("$", "")):
        s = raw.strip()
        if not s:
            continue
        is_product_line = re.search(r"profit|margin|earn|contribution", s, re.I)
        has_hint = re.search(r"have|available|in stock|on hand|capacity|capacities|at most|up to|limited|limit|maximum|total|only|supply\b", s, re.I)
        if is_product_line and not has_hint:
            continue
        if not has_hint and ":" not in s and not re.search(r"<=?", s):
            continue
        for m in re.finditer(
            r"([A-Za-z][A-Za-z ]{1,30}?)\s+(?:capacity|available|in stock|on hand)\s*(?:of\s*|is\s*|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:units?|hours?|hrs?|kg|tons?)?",
            s, re.I,
        ):
            name = _clean_name(re.sub(r"^(we|they|factory|we have|available|total|capacity of|capacity)\s+", "", m.group(1), flags=re.I))
            if not name or name.lower() in STOP_WORDS or re.search(r"profit|margin|demand|supply", name, re.I):
                continue
            if len(name) > 40:
                continue
            caps.setdefault(name, float(m.group(2)))
        for m in re.finditer(
            r"([A-Za-z][A-Za-z ]{1,30}?)\s*(?:capacity|available|in stock|on hand)?\s*(?:<=|<|:|=|is|of)\s*(\d+(?:\.\d+)?)\s*(?:units?|hours?|hrs?|kg|tons?)?",
            s, re.I,
        ):
            name = _clean_name(re.sub(r"^(we|they|factory|we have|available|total|capacity of|capacity)\s+", "", m.group(1), flags=re.I))
            if not name or name.lower() in STOP_WORDS or re.search(r"profit|margin|demand|supply", name, re.I):
                continue
            if len(name) > 40:
                continue
            caps[name] = float(m.group(2))
        for m in re.finditer(
            r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*(units?|hours?|hrs?|hr|kg|kgs|tons?|tonnes?|days?)?\s*(?:of\s+)?([A-Za-z][A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|$|available|in stock|capacity|at most|maximum|total))",
            s, re.I,
        ):
            if re.search(r"profit|margin|demand|supply|product", m.group(3), re.I):
                continue
            name = _clean_name(m.group(3))
            if not name or name.lower() in STOP_WORDS or len(name) > 40:
                continue
            caps.setdefault(name, float(m.group(1)))
        for key, value in _parse_resource_values(s).items():
            if re.search(r"profit|demand|supply", key, re.I):
                continue
            caps.setdefault(key, value)
    cap_match = re.search(r"We have (.*?)(?:Demand|$)", text, re.I)
    if cap_match:
        for key, value in _parse_resource_values(cap_match.group(1)).items():
            caps.setdefault(key, value)
    return caps


def _parse_demand_free(text: str, product_label: str) -> float | None:
    label = re.sub(r"^product\s+", "", product_label, flags=re.I).strip()
    last_word = (label.split() or [label])[-1]
    base = [product_label, label, last_word]
    cands: list[str] = []
    for b in base:
        if not b:
            continue
        cands.append(b)
        # plural/singular tolerant: "cake" also matches "cakes"
        if re.search(r"s$", b, re.I) and len(b) > 3:
            cands.append(b[:-1])
        else:
            cands.append(b + "s")
    for cand in cands:
        e = re.escape(cand)
        for pattern in [
            rf"demand(?:\s+for)?\s+{e}\s*(?:is\s+)?(?:at most|max(?:imum)?|cannot exceed|up to|of|is|:|=|<=?)\s*(\d+(?:\.\d+)?)",
            rf"{e}\s*(?:demand)?\s*(?:at most|max(?:imum)?|cannot exceed|up to|limit(?:ed to)?|<=|<)\s*(\d+(?:\.\d+)?)",
            rf"(?:at most|max(?:imum)?|up to|no more than)\s*(\d+(?:\.\d+)?)\s*(?:units?\s+)?(?:of\s+)?{e}",
        ]:
            m = re.search(pattern, text, re.I)
            if m:
                return float(m.group(1))
    return None


def _split_product_clauses(sentence: str) -> list[str]:
    """Split "Profit per X1 is 40, profit per X2 is 30" into per-product clauses."""
    parts = re.split(
        r",(?=\s*(?:and\s+)?(?:profit|margin|each|every|product\b))|\s+and\s+(?=[A-Za-z][A-Za-z0-9\- ]{0,30}?\s+for\s+a\s+(?:profit|margin))",
        sentence, flags=re.I,
    )
    if len(parts) <= 1:
        return [sentence]
    with_profit = [p for p in parts if _profit_from_sentence(p) is not None]
    if len(with_profit) > 1:
        return [p.strip() for p in parts if p.strip()]
    return [sentence]


def _attach_split_needs(clean: str, products: list[dict]) -> None:
    """Attach "X1 requires 2 machine hours..." lines to profit-only products by name."""
    needy = [p for p in products if not p["resource_usage"]]
    if not needy:
        return
    for raw in re.split(r"[.\n;]+", clean):
        sentence = raw.strip()
        if not sentence or re.search(r"profit|margin|earn|contribution|gain|revenue", sentence, re.I):
            continue
        need_match = re.match(
            r"\s*(?:for\s+)?([A-Za-z][A-Za-z0-9\-]*)\s+(?:needs?|uses?|requires?|consumes?|takes?|takes up|spends?)\s+(.*)",
            sentence, re.I,
        )
        if not need_match:
            continue
        name_raw, usage_text = need_match.groups()
        usage = _usage_free_form(usage_text, None)
        if not usage:
            continue
        target = next((p for p in needy if _canon_hit(p["name"], _clean_name(name_raw))), None)
        if target:
            target["resource_usage"] = usage


def _canon_hit(a: str, b: str) -> bool:
    ka, kb = _canon(a), _canon(b)
    if not ka or not kb:
        return bool(ka and ka == kb)
    if ka == kb:
        return True
    if len(kb) >= 3 and kb in ka:
        return True
    if len(ka) >= 3 and ka in kb:
        return True
    return False


def _parse_production_free(text: str):
    clean = text.replace("$", "")
    products: list[dict] = []
    auto_idx = 0
    for raw in re.split(r"[.\n;]+", clean):
        sentence = raw.strip()
        if not sentence:
            continue
        for clause in _split_product_clauses(sentence):
            if not re.search(r"profit|margin|earn|contribution|gain|revenue", clause, re.I):
                continue
            if re.match(r"(?i)^(we have|available|in stock|capacity|total)", clause) and not re.search(r"needs?|uses?|requires?|consumes?|takes?|gives?|earns?|yields?|makes?", clause, re.I):
                continue
            profit = _profit_from_sentence(clause)
            if profit is None or profit < 0:
                continue
            auto_idx += 1
            name = _product_name_from_sentence(clause, f"Product {chr(64 + auto_idx)}")
            need_match = re.search(r"(?:needs?|uses?|requires?|consumes?|takes?|takes up|spends?)\s+(.*)", clause, re.I)
            usage = _usage_free_form(need_match.group(1) if need_match else clause, profit)
            if not usage:
                # profit-only clause (usage lives in another sentence) — remember for phase 2
                if not any(_canon(p["name"]) == _canon(name) for p in products):
                    products.append({"name": name, "profit": profit, "resource_usage": {}, "demand_limit": None})
                continue
            if any(_canon(p["name"]) == _canon(name) for p in products):
                continue
            products.append({"name": name, "profit": profit, "resource_usage": usage, "demand_limit": None})
    # Phase 2: attach split needs-lines, then drop profit-only entries without usage.
    _attach_split_needs(clean, products)
    products = [p for p in products if p["resource_usage"]]
    if not products:
        return None
    capacities = _parse_capacities_free(clean)
    for product in products:
        for res in list(product["resource_usage"]):
            if res in capacities:
                continue
            hit = next((c for c in capacities if _canon_hit(c, res)), None)
            if hit:
                product["resource_usage"][hit] = product["resource_usage"].pop(res)
    for product in products:
        product["demand_limit"] = _parse_demand_free(clean, product["name"])
    used = {_canon(r) for p in products for r in p["resource_usage"]}
    filtered = {k: v for k, v in capacities.items() if _canon(k) in used}
    final_caps = filtered or capacities
    if not products or not final_caps:
        return None
    return {"products": products, "capacities": final_caps}


def _extract_with_fallback(problem: str) -> BusinessModel:
    text = problem.replace("$", "")
    game_match = re.search(r"choose\s+([A-Za-z]+)\s+or\s+([A-Za-z]+)", text, re.I)
    if game_match and re.search(r"player\s+1|player\s+2|payoff|expected payoff|game|versus|\bvs\b", text, re.I):
        first_action, second_action = game_match.groups()
        listed_payoffs = re.search(
            r"payoffs?\s+(?:are|of|is)?\s*\[?(-?\d+(?:\.\d+)?)\s*(?:,\s*and|,|and|;|\s)\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*and|,|and|;|\s)\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*and|,|and|;|\s)\s*(-?\d+(?:\.\d+)?)\]?",
            text,
            re.I,
        )
        if listed_payoffs:
            values = [float(value) for value in listed_payoffs.groups()]
            payoffs = {
                first_action: {first_action: values[0], second_action: values[1]},
                second_action: {first_action: values[2], second_action: values[3]},
            }
            return BusinessModel.model_validate({
                "model_type": "game_theory",
                "model": {"player_one_actions": [first_action, second_action], "player_two_actions": [first_action, second_action], "payoffs": payoffs, "zero_sum": True},
                "problem_summary": "Choose a mixed strategy for Player 1 against a competing Player 2.",
                "technique": "Zero-sum game linear programming",
                "objective": "Maximize Player 1's guaranteed expected payoff",
                "variables": [f"probability of {first_action}", f"probability of {second_action}"],
                "constraints": ["strategy probabilities sum to 1", "expected payoff is at least the game value against every opposing action"],
                "assumptions": ["The four listed payoffs are ordered by Player 1 action rows, then Player 2 action columns."],
            })
        payoff_patterns = {
            (first_action, first_action): rf"both\s+{re.escape(first_action)}.*?(-?\d+(?:\.\d+)?)",
            (first_action, second_action): rf"Player\s+1\s+{re.escape(first_action)}.*?Player\s+2\s+{re.escape(second_action)}.*?(-?\d+(?:\.\d+)?)",
            (second_action, first_action): rf"Player\s+1\s+{re.escape(second_action)}.*?Player\s+2\s+{re.escape(first_action)}.*?(-?\d+(?:\.\d+)?)",
            (second_action, second_action): rf"both\s+{re.escape(second_action)}.*?(-?\d+(?:\.\d+)?)",
        }
        payoffs = {}
        ok = True
        for (player_one_action, player_two_action), pattern in payoff_patterns.items():
            match = re.search(pattern, text, re.I)
            if not match:
                ok = False
                break
            payoffs.setdefault(player_one_action, {})[player_two_action] = float(match.group(1))
        if ok:
            return BusinessModel.model_validate({
                "model_type": "game_theory",
                "model": {"player_one_actions": [first_action, second_action], "player_two_actions": [first_action, second_action], "payoffs": payoffs, "zero_sum": True},
                "problem_summary": "Choose a mixed strategy for Player 1 against a competing Player 2.",
                "technique": "Zero-sum game linear programming",
                "objective": "Maximize Player 1's guaranteed expected payoff",
                "variables": [f"probability of {first_action}", f"probability of {second_action}"],
                "constraints": ["strategy probabilities sum to 1", "expected payoff is at least the game value against every opposing action"],
                "assumptions": ["The game is zero-sum and both players choose actions simultaneously."],
            })
    # Free-form transportation attempt (supplies + demands + costs in any wording).
    transport = _parse_transportation_free(text)
    if transport is not None:
        sources, destinations, costs = transport
        return BusinessModel.model_validate({
            "model_type": "transportation",
            "model": {
                "sources": [{"name": name, "supply": qty} for name, qty in sources],
                "destinations": [{"name": name, "demand": qty} for name, qty in destinations],
                "costs": costs,
            },
            "problem_summary": "Choose shipment quantities to minimize total shipping cost.",
            "technique": "Transportation linear programming",
            "objective": "Minimize total shipping cost",
            "variables": [f"quantity from {s} to {d}" for s, _ in sources for d, _ in destinations],
            "constraints": [f"shipments from {s} must not exceed supply {q}" for s, q in sources]
            + [f"deliveries to {d} must meet demand {q}" for d, q in destinations],
            "assumptions": ["Understood your wording automatically; shipping cost is linear in quantity."],
        })
    # Minimum-cost purchase / grocery / diet attempt — solved directly, never relabelled.
    procurement = _parse_procurement_free(text)
    if procurement is not None:
        shops, items, requirements, prices, discounts = procurement
        return BusinessModel.model_validate({
            "model_type": "procurement",
            "model": {
                "shops": shops,
                "items": items,
                "requirements": requirements,
                "prices": prices,
                "discounts": [{"shop": s, "amount": a, "threshold": t} for s, a, t in discounts],
            },
            "problem_summary": "Choose where to buy each item to meet all requirements at the lowest total cost.",
            "technique": "Linear programming (minimum-cost purchase plan)",
            "objective": "Minimize total purchase cost",
            "variables": [f"quantity of {item} from {shop}" for shop in shops for item in items],
            "constraints": [f"buy at least {requirements[item]:g} {item}" for item in items]
            + [f"{shop} gives {amount:g} off when spending reaches {threshold:g} there" for shop, amount, threshold in discounts],
            "assumptions": ["Understood your wording automatically; unit prices are fixed and quantities may be fractional."],
        })
    # Legacy strict production path first (keeps old prompts working).
    products = []
    for sentence in text.split("."):
        entry = re.search(r"(Product\s+[A-Za-z0-9]+).*?(\d+(?:\.\d+)?)\s+profit.*?(?:needs|uses|requires|consumes)\s+(.*)", sentence, re.I)
        if not entry:
            continue
        name, profit, resource_text = entry.groups()
        segment = f"{profit} {resource_text}"
        usage = _parse_resource_values(segment)
        demand_match = re.search(rf"Demand for {re.escape(name.split()[-1])}.*?(?:at most|maximum|max)\s*(\d+(?:\.\d+)?)", text, re.I)
        if usage:
            products.append({"name": name, "profit": float(profit), "resource_usage": usage, "demand_limit": float(demand_match.group(1)) if demand_match else None})

    capacities = _parse_resource_values(re.search(r"We have (.*?)(?:Demand|$)", text, re.I).group(1) if re.search(r"We have (.*?)(?:Demand|$)", text, re.I) else "")
    if not capacities:
        capacities = _parse_capacities_free(text)
    for product in products:
        for resource in list(product["resource_usage"]):
            matching_capacity = next(
                (name for name in capacities if name.lower() in resource.lower() or resource.lower() in name.lower()),
                None,
            )
            if matching_capacity and resource not in capacities:
                capacities[resource] = capacities.pop(matching_capacity)
    if products and capacities:
        return BusinessModel.model_validate({
            "model_type": "production",
            "model": {"products": products, "capacities": capacities},
            "problem_summary": "Choose production quantities to maximize total profit under resource limits.",
            "technique": "Linear programming",
            "objective": "Maximize total profit",
            "variables": [f"quantity of {product['name']}" for product in products],
            "constraints": [f"resource usage must not exceed {name} capacity" for name in capacities],
            "assumptions": ["Understood your wording automatically; quantities may be fractional unless whole units are required."],
        })
    # Free-form production path: any wording with the same numbers.
    free = _parse_production_free(text)
    if free is not None:
        return BusinessModel.model_validate({
            "model_type": "production",
            "model": free,
            "problem_summary": "Choose production quantities to maximize total profit under resource limits.",
            "technique": "Linear programming",
            "objective": "Maximize total profit",
            "variables": [f"quantity of {product['name']}" for product in free["products"]],
            "constraints": [f"resource usage must not exceed {name} capacity" for name in free["capacities"]],
            "assumptions": ["Understood your wording automatically; quantities may be fractional unless whole units are required."],
        })
    if re.search(r"solve\s+(for\s+)?x\b|derivative|integral|quadratic|differentiate|integrate|simplify", text, re.I):
        raise ValueError("OptiSolve only solves optimization problems — production planning, transportation, shopping/purchase planning, and game theory — not general math. Describe profits with limited resources, shipping supplies/demands/costs, shopping needs with shop prices, or a game payoff, in any wording.")
    if re.search(r"warehouse|ship|shipping|transport|freight|supplier|customer|distribution|supply|demand", text, re.I):
        partial = _diagnose_transportation(text)
        if partial["sources"] and partial["destinations"] and partial["missing_routes"]:
            missing = ", ".join(f"{s} to {d} costs ?" for s, d in partial["missing_routes"][:4])
            first_s, first_d = partial["missing_routes"][0]
            found = (
                f"sources {', '.join(f'{n} ({q:g})' for n, q in partial['sources'])}; "
                f"destinations {', '.join(f'{n} ({q:g})' for n, q in partial['destinations'])}; "
                f"costs {partial['have']}/{partial['needed']} found"
            )
            raise ValueError(
                f"Got it — this is a shipping problem and I understood most of it ({found}). "
                f"I'm still missing {len(partial['missing_routes'])} route cost(s): {missing}. "
                f"Just add them in your own words — e.g. \"{first_s} to {first_d} costs 5\" — and I'll solve it instantly."
            )
        if partial["sources"] or partial["destinations"]:
            bits = []
            if partial["sources"]:
                bits.append("sources " + ", ".join(f"{n} ({q:g})" for n, q in partial["sources"]))
            if partial["destinations"]:
                bits.append("destinations " + ", ".join(f"{n} ({q:g})" for n, q in partial["destinations"]))
            raise ValueError(
                f"Got it — this is a shipping problem ({'; '.join(bits)}). "
                "To solve it I still need supplies, demands, and every route cost — e.g. "
                "'Warehouse A supplies 100, Store X needs 120, A to X costs 4'. Add those in any wording and try again."
            )
        raise ValueError(
            "Got it — you want to optimize shipping. Tell me in your own words: (1) who supplies how much, "
            "(2) who needs how much, (3) what each route costs. Example: "
            "'Warehouse A has 100 units, Store X needs 120 units, shipping from A to X costs 4 per unit.' "
            "Any wording with the same numbers works."
        )
    if re.search(r"shop|store|vendor|sells?|purchase|buys?|prices?|discount|grocery|mart", text, re.I) and not re.search(r"profit|margin|contribution", text, re.I):
        shops: list[str] = []
        seen: set[str] = set()
        for m in re.finditer(SHOP_WORD + r"\s+([A-Za-z0-9]+)", text, re.I):
            name = _clean_name(m.group(0))
            key = _canon(name)
            if name and key not in seen:
                seen.add(key)
                shops.append(name)
        needs = [f"{m.group(2).strip()} ({m.group(1)})" for m in re.finditer(r"(?:at\s+least\s+|needs?\s+(?:at\s+least\s+)?)(\d+(?:\.\d+)?)\s*(?:kg|liters?|units?)?\s*(?:of\s+)?([A-Za-z][A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|\.|per\s+(?:month|day)|$))", text, re.I) if "₹" not in m.group(0) and "$" not in m.group(0)]
        if shops or needs:
            bits = []
            if shops:
                bits.append("shops " + ", ".join(shops))
            if needs:
                bits.append("needs " + ", ".join(needs[:5]))
            raise ValueError(
                f"Got it — this is a shopping / purchase problem ({'; '.join(bits)}). "
                "To find your cheapest plan, add how much you need and the price at each shop — e.g. "
                "'We need at least 10 kg of rice. Shop A sells rice at ₹50/kg, Shop B at ₹45/kg.' Any wording works."
            )
    if re.search(r"profit|margin|earn|contribution|gain|revenue|produces?|manufactures?|capacity|available|machine|labo?r|wood|oven", text, re.I):
        partial = _diagnose_production(text)
        if partial["products"] or partial["capacities"]:
            bits = []
            if partial["products"]:
                bits.append("products " + ", ".join(partial["products"]))
            if partial["capacities"]:
                bits.append("limits " + ", ".join(f"{k} ({v:g})" for k, v in partial["capacities"].items()))
            first = partial["products"][0] if partial["products"] else "chairs"
            singular = first[:-1] if re.search(r"[^s]s$", first, re.I) and len(first) > 3 else first
            raise ValueError(
                f"Got it — this is a production problem ({'; '.join(bits)}). "
                f"To find your optimal plan, add what each item needs and what limits you — e.g. "
                f"\"Each {singular} needs 2 wood and 1 labor. We have 100 wood and 80 labor.\" Any wording works."
            )
    raise ValueError("Tell me about your problem in your own words and I'll work out the rest — e.g.: 'We make chairs (40 profit, 2 wood + 1 labor) and tables (30 profit, 1 wood + 2 labor). We have 100 wood and 80 labor.' Or shipping: 'Warehouse A has 100, Store X needs 120, A to X costs 4.' Or shopping: 'We need 10 kg rice. Shop A sells rice at ₹50/kg, Shop B at ₹45/kg.' I handle production, transportation, shopping/purchase, and game-theory problems (not general math).")


def _diagnose_transportation(text: str) -> dict:
    """Best-effort trace of shipping intent for friendly clarification messages."""
    sources: list[tuple[str, float]] = []
    destinations: list[tuple[str, float]] = []
    seen: set[str] = set()

    def push(items: list, name: str, qty: float, kind: str):
        cleaned = _clean_name(name)
        if not cleaned:
            return
        key = kind + ":" + re.sub(r"\s+", " ", cleaned.lower())
        if key in seen:
            return
        seen.add(key)
        items.append((cleaned, qty))

    for pattern in [
        r"([A-Za-z][\w\- ]*?)\s+(?:has\s+supply|suppl\w*(?:ies)?|supply)\s+(?:of\s+)?(\d+(?:\.\d+)?)",
        r"(?:supply|stock|inventory|capacity|available)\s+(?:at|from|of)\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)",
        r"(factory|warehouse|plant|supplier|depot|origin)\s+([A-Za-z0-9\- ]*?)\s*(?:has|holds?|stocks?|with)?\s*(?:supply|stock|inventory|capacity|available|units?)?\s*(?:of\s*|is\s*|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:units?)?",
        r"([A-Za-z][\w\- ]*?)\s+(?:needs?|requires?|demands?|orders?|wants?)\s+(\d+(?:\.\d+)?)",
    ]:
        for m in re.finditer(pattern, text, re.I):
            if "needs" in m.group(0).lower() or "require" in m.group(0).lower() or "demand" in m.group(0).lower() or "order" in m.group(0).lower() or "want" in m.group(0).lower():
                if re.search(r"profit|margin|machine|labo?r|oven|wood|chips", m.group(0), re.I):
                    continue
                push(destinations, m.group(1), float(m.group(2)), "d")
            elif len(m.groups()) == 3:
                push(sources, f"{m.group(1)} {m.group(2)}".strip(), float(m.group(3)), "s")
            else:
                push(sources, m.group(1), float(m.group(2)), "s")
    # route costs found (loose last-word match so "A"/"Warehouse A" align)
    costs_found: set[tuple[str, str]] = set()
    for m in re.finditer(
        r"([A-Za-z][\w\- ]*?)\s*(?:->|to)\s*([A-Za-z][\w\- ]*?)\s*(?:costs?|is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)",
        text, re.I,
    ):
        s_tok = re.sub(r"\s+", " ", m.group(1).strip().lower())
        d_tok = re.sub(r"\s+", " ", m.group(2).strip().lower())
        for s, _ in sources:
            for d, _ in destinations:
                s_last = re.sub(r"\s+", " ", s.strip().lower()).split(" ")[-1]
                d_last = re.sub(r"\s+", " ", d.strip().lower()).split(" ")[-1]
                if s_last == s_tok.split(" ")[-1] and d_last == d_tok.split(" ")[-1]:
                    costs_found.add((s, d))
    missing_routes = [(s, d) for s, _ in sources for d, _ in destinations if (s, d) not in costs_found]
    needed = len(sources) * len(destinations)
    return {"sources": sources, "destinations": destinations, "missing_routes": missing_routes,
            "have": len(costs_found), "needed": needed}


def _diagnose_production(text: str) -> dict:
    """Best-effort trace of production intent for friendly clarification messages."""
    products: list[str] = []
    clean = text.replace("$", "")
    for raw in re.split(r"[.\n;]+", clean):
        sentence = raw.strip()
        if not sentence or not re.search(r"profit|margin|earn|contribution|gain|revenue", sentence, re.I):
            continue
        try:
            name = _product_name_from_sentence(sentence, "")
        except Exception:
            name = ""
        if name and name not in products:
            products.append(name)
    try:
        capacities = _parse_capacities_free(clean)
    except Exception:
        capacities = {}
    return {"products": products[:5], "capacities": dict(list(capacities.items())[:6])}


SHOP_WORD = r"(?:shop|store|vendor|supplier|mart|market|mall|outlet|dealer)"
QTY_UNIT = r"(?:kg|kgs|kilos?|grams?|gms?|g|liters?|litres?|ltrs?|l|ml|units?|packs?|packets?|bottles?|cans?|pieces?|dozens?|bags?)"
MONEY = r"(?:₹|Rs\.?|\$|€|£)"


def _clean_item_name(raw: str) -> str:
    s = _clean_name(raw)
    s = re.sub(r"\s+(per\s+(?:month|day|week|year)|each\s+(?:month|day)|monthly|daily)$", "", s, flags=re.I).strip()
    s = re.sub(r"\s+from\s+(?:shop|store|vendor|supplier|mart|market|mall|outlet|dealer)\s+[A-Za-z0-9]+$", "", s, flags=re.I).strip()
    s = re.sub(r"^(at\s+least|atleast|minimum\s+of|minimum|min)\s+", "", s, flags=re.I).strip()
    return s


def _parse_procurement_free(text: str):
    """Minimum-cost purchase plans in any wording. Returns
    (shops, items, requirements, prices, discounts) or None."""
    if not re.search(SHOP_WORD + r"|sells?|purchase|buys?|prices?|discount|grocery", text, re.I):
        return None
    if re.search(r"profit|margin|contribution", text, re.I):
        return None
    shops: list[str] = []
    seen_shops: set[str] = set()
    for m in re.finditer(SHOP_WORD + r"\s+([A-Za-z0-9]+)", text, re.I):
        name = _clean_name(m.group(0))
        key = _canon(name)
        if not name or key in seen_shops:
            continue
        seen_shops.add(key)
        shops.append(name)
    if not shops:
        return None

    def shop_mentions(sentence: str) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for m in re.finditer(SHOP_WORD + r"\s+([A-Za-z0-9]+)", sentence, re.I):
            cand = _clean_name(m.group(0))
            hit = next((s for s in shops if _canon_hit(s, cand)), None)
            if hit:
                out.append((m.start(), hit))
        return out

    def shop_in_sentence(sentence: str) -> str | None:
        found = shop_mentions(sentence)
        return found[0][1] if found else None

    items: list[str] = []
    requirements: dict[str, float] = {}
    seen_items: set[str] = set()

    def add_req(raw_item: str, qty: float):
        item = _clean_item_name(raw_item)
        if not item or len(item) > 40:
            return
        if re.search(r"demand|supply|profit|discount|purchase\s+value|^total$", item, re.I):
            return
        key = _canon(item)
        if not key or key in seen_items:
            return
        seen_items.add(key)
        items.append(item)
        requirements[item] = qty

    req_look = r"(?=\s*(?:and|,|;|\.|per\s+(?:month|day|week)|each\s+(?:month|day)|$))"
    sentences = re.split(r"[.\n;]+", text)
    for raw in sentences:
        s = raw.strip()
        if not s:
            continue
        for m in re.finditer(
            r"(?:needs?|requires?|must\s+(?:buy|get|have)|should\s+(?:buy|get))\s+(?:at\s+least\s+|atleast\s+|a\s+minimum\s+of\s+|minimum\s+of\s+|minimum\s+|min\s+)?(\d+(?:\.\d+)?)\s*" + QTY_UNIT + r"?\s*(?:of\s+)?([A-Za-z][A-Za-z][\w\- ]*?)" + req_look,
            s, re.I,
        ):
            if re.search(MONEY, m.group(0)):
                continue
            add_req(m.group(2), float(m.group(1)))
        for m in re.finditer(r"at\s+least\s+(\d+(?:\.\d+)?)\s*" + QTY_UNIT + r"?\s*(?:of\s+)?([A-Za-z][A-Za-z][\w\- ]*?)" + req_look, s, re.I):
            if re.search(MONEY, m.group(0)):
                continue
            add_req(m.group(2), float(m.group(1)))
        if re.search(r"needs?|requires?|at\s+least|minimum", s, re.I) and requirements:
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*" + QTY_UNIT + r"\s*(?:of\s+)?([A-Za-z][A-Za-z][\w\- ]*?)" + req_look, s, re.I):
                if re.search(MONEY, m.group(0)):
                    continue
                add_req(m.group(2), float(m.group(1)))
    if not items:
        return None

    prices: dict[str, dict[str, float]] = {}
    last_shop: str | None = None
    for raw in sentences:
        s = raw.strip()
        if not s:
            continue
        mentions = shop_mentions(s)
        mentioned = mentions[0][1] if mentions else None
        if mentioned:
            last_shop = mentioned
        if re.search(r"respectively", s, re.I) and mentioned:
            nums = [float(v) for v in re.findall(r"(?:" + MONEY + r")?\s*(\d+(?:\.\d+)?)", s)]
            if len(nums) == len(items):
                prices.setdefault(mentioned, {}).update({item: value for item, value in zip(items, nums)})
                last_shop = mentioned
                continue
        last_item: str | None = None
        for m in re.finditer(r"([A-Za-z][A-Za-z][\w\- ]*?)\s+at\s+(" + MONEY + r")?\s*(\d+(?:\.\d+)?)\s*(\/\s*\w+)?", s, re.I):
            if not m.group(2) and not m.group(4):
                continue
            cand = _clean_item_name(m.group(1))
            # shop restated inside the price ("Shop B at ₹45")? it owns this price
            inner = next((s for s in shops if _canon_hit(s, _clean_name(m.group(1)))), None)
            shop = inner or next((sh for pos, sh in mentions if pos < m.start()), None) or mentioned or last_shop
            if not shop:
                continue
            hit = next((it for it in items if _canon_hit(it, cand)), None)
            if not hit:
                # "Shop B at ₹45" — shop restated without the item: same item as before
                if inner and last_item:
                    hit = last_item
                else:
                    continue
            last_item = hit
            prices.setdefault(shop, {}).setdefault(hit, float(m.group(3)))
        if mentions:
            last_shop = mentions[-1][1]
    priced_shops = [sh for sh in shops if prices.get(sh)]
    if not priced_shops:
        return None
    for item in items:
        if not any(item in prices.get(sh, {}) for sh in priced_shops):
            return None

    discounts: list[tuple[str, float, float]] = []
    for m in re.finditer(
        SHOP_WORD + r"\s+([A-Za-z0-9]+)\s*[^.\n;]*?(\d+(?:\.\d+)?)\s*discount[^.\n;]*?(?:at\s+least|over|above|minimum\s+(?:of\s+)?|more\s+than|exceeding|reaches|reaching)\s*(?:" + MONEY + r")?\s*(\d+(?:\.\d+)?)",
        text, re.I,
    ):
        cand = _clean_name(m.group(0))
        hit = next((s for s in priced_shops if _canon_hit(s, cand)), priced_shops[0])
        amount, threshold = float(m.group(2)), float(m.group(3))
        if amount > 0 and threshold > 0 and not any(d[0] == hit for d in discounts):
            discounts.append((hit, amount, threshold))
    return priced_shops, items, requirements, prices, discounts


def _parse_transportation_free(text: str):
    # Production texts ("Demand for A at most 40", "We have 100 units Machine")
    # must never be mistaken for shipping.
    if re.search(r"profit|margin|earn|contribution|gain|revenue", text, re.I) and not re.search(
        r"warehouse|factory|plant|supplier|depot|store|shop|ship|freight|transport|deliver|customer|retailer", text, re.I
    ):
        return None
    sources: list[tuple[str, float]] = []
    destinations: list[tuple[str, float]] = []
    seen: set[str] = set()

    def push(items: list, name: str, qty: float, kind: str):
        cleaned = _clean_name(name)
        if not cleaned:
            return
        if re.match(r"^(we|they|you|he|she|it|there|here|this|that|they have|we have)$", cleaned, re.I):
            return
        # lazy captures like "I have a warehouse A" (from "...with 100 units")
        # are not places — the keyword pattern catches the real place
        if re.search(r"\b(have|has|with|and|or)\b", cleaned, re.I) and not re.match(
            r"^(factory|warehouse|plant|supplier|depot|origin|store|shop|city|customer)\b", cleaned, re.I
        ):
            return
        key = kind + ":" + re.sub(r"\s+", " ", cleaned.lower())
        if key in seen:
            return
        seen.add(key)
        items.append((cleaned, qty))

    for pattern in [
        r"([A-Za-z][\w\- ]*?)\s+(?:has\s+supply|suppl\w*(?:ies)?|supply)\s+(?:of\s+)?(\d+(?:\.\d+)?)",
        r"(?:supply|stock|inventory|capacity|available)\s+(?:at|from|of)\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)",
        r"([A-Za-z][\w\- ]*?)\s*(?:can\s+supply|can\s+ship|can\s+send|ships?|sends?|provides?|offers?)\s+(?:up\s+to\s+|about\s+)?(\d+(?:\.\d+)?)\s*(?:units?)?",
        r"([A-Za-z][\w\- ]*?)\s*(?:has|holds?|stocks?|with|contains?)\s+(?:up\s+to\s+|about\s+|around\s+)?(\d+(?:\.\d+)?)\s*units?\s*(?:supply|stock|inventory|available|to\s+ship|to\s+send)?",
    ]:
        for m in re.finditer(pattern, text, re.I):
            push(sources, m.group(1), float(m.group(2)), "s")
    for m in re.finditer(
        r"(factory|warehouse|plant|supplier|depot|origin)\s+([A-Za-z0-9\- ]*?)\s*(?:has|holds?|stocks?|with|can supply|produces?|makes?|offers?)?\s*(?:supply|stock|inventory|capacity|available|units?)?\s*(?:of\s*|is\s*|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:units?)?",
        text, re.I,
    ):
        push(sources, f"{m.group(1)} {m.group(2)}".strip(), float(m.group(3)), "s")
    for pattern in [
        r"([A-Za-z][\w\- ]*?)\s+(?:needs?\s+demand|requires?\s+demand|has\s+demand|demand)\s+(?:of\s+|is\s+)?(\d+(?:\.\d+)?)",
        r"(?:demand|need|needs|requirement)\s+(?:at|for|of|from)\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)",
    ]:
        for m in re.finditer(pattern, text, re.I):
            if re.search(r"demand for", m.group(0), re.I) and re.search(r"at most|maximum|max", text, re.I):
                continue
            push(destinations, m.group(1), float(m.group(2)), "d")
    for m in re.finditer(
        r"(store|shop|city|customer|retailer|destination|market|client)\s+([A-Za-z0-9\- ]*?)\s*(?:needs?|requires?|demands?|wants?|orders?|takes?)?\s*(?:demand|units?|need)?\s*(?:of\s*|is\s*|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:units?)?",
        text, re.I,
    ):
        push(destinations, f"{m.group(1)} {m.group(2)}".strip(), float(m.group(3)), "d")
    # generic "X needs 120" (but not profit lines, and not "X needs demand N")
    for m in re.finditer(r"([A-Za-z][\w\- ]*?)\s+(?:needs?|requires?|demands?|orders?|wants?|takes?|receives?)\s+(\d+(?:\.\d+)?)\s*(?:units?)?(?!\s*(?:profit|margin))", text, re.I):
        if re.search(r"at most|maximum|\bmax\b|up to|no more than|cannot exceed|\blimit", m.group(0), re.I):
            continue
        if re.search(r"profit|margin|machine|labo?r|oven|wood|chips", m.group(0), re.I):
            continue
        if re.search(r"\b(needs?|requires?|demands?|demand|orders?)\s*$", m.group(1), re.I):
            continue
        push(destinations, m.group(1), float(m.group(2)), "d")
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*units?\s+(?:for|to|at)\s+([A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|\.|$))", text, re.I):
        if re.search(r"profit|margin|machine|labo?r|oven|wood|chips", m.group(0), re.I):
            continue
        push(destinations, m.group(2), float(m.group(1)), "d")
    src_norms = {re.sub(r"\s+", " ", s.lower()) for s, _ in sources}
    destinations = [(n, q) for n, q in destinations if re.sub(r"\s+", " ", n.lower()) not in src_norms]
    if not sources or not destinations:
        return None
    src_names = [s for s, _ in sources]
    dst_names = [d for d, _ in destinations]

    def match_name(token: str, known: list[str]) -> str | None:
        t = re.sub(r"\s+", " ", token.strip().lower())
        for k in known:
            if re.sub(r"\s+", " ", k.strip().lower()) == t:
                return k
        for k in known:
            if re.sub(r"\s+", " ", k.strip().lower()).endswith(" " + t):
                return k
        singles = [k for k in known if (re.sub(r"\s+", " ", k.strip().lower()).split(" ") or [""])[-1] == t]
        if len(singles) == 1:
            return singles[0]
        contains = [k for k in known if t in re.sub(r"\s+", " ", k.strip().lower()) or re.sub(r"\s+", " ", k.strip().lower()) in t]
        if len(contains) == 1:
            return contains[0]
        # fuzzy: strip leading words ("shipping from A" -> "A") and retry each suffix
        words = [w for w in t.split(" ") if w and not re.match(r"^(shipping|shipment|freight|transport|transportation|delivery|deliver|costs?|costing|from|the|unit|units|per)$", w)]
        for i in range(len(words)):
            suffix = " ".join(words[i:])
            for k in known:
                if re.sub(r"\s+", " ", k.strip().lower()) == suffix:
                    return k
            for k in known:
                if re.sub(r"\s+", " ", k.strip().lower()).endswith(" " + suffix):
                    return k
            short = [k for k in known if (re.sub(r"\s+", " ", k.strip().lower()).split(" ") or [""])[-1] == suffix]
            if len(short) == 1:
                return short[0]
        return None

    costs: dict[str, dict[str, float]] = {}
    for pattern in [
        r"([A-Za-z][\w\- ]*?)\s*(?:->|→|—|–|-)\s*([A-Za-z][\w\- ]*?)\s*(?:costs?|costing|price|rate|freight|charge)?\s*(?:is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)(?:\s*per unit)?",
        r"([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)\s*(?:costs?|costing|price|rate|freight|charge|fare)?\s*(?:is|:|=|of)?\s*\$?\s*(\d+(?:\.\d+)?)(?:\s*per unit)?",
        r"(?:cost|price|rate|freight|charge|fare)\s+(?:from|of|for)\s+([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)(?:\s*per unit)?",
        r"from\s+([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)\s*(?:costs?|is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)(?:\s*per unit)?",
    ]:
        for m in re.finditer(pattern, text, re.I):
            s = match_name(m.group(1), src_names)
            d = match_name(m.group(2), dst_names)
            if s and d:
                costs.setdefault(s, {}).setdefault(d, float(m.group(3)))
    # "$4 from A to X" word order
    for m in re.finditer(
        r"\$?\s*(\d+(?:\.\d+)?)\s*(?:per unit\s+)?from\s+([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|\.|$))",
        text, re.I,
    ):
        s = match_name(m.group(2), src_names)
        d = match_name(m.group(3), dst_names)
        if s and d:
            costs.setdefault(s, {}).setdefault(d, float(m.group(1)))
    needed = len(sources) * len(destinations)
    have = sum(len(row) for row in costs.values())
    if have < needed:
        return None
    return sources, destinations, costs


def _parse_resource_values(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for value, _, resource_before_unit, resource_after_unit, _ in re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:(hours?|kg|units?)\s+(?:on|of)?\s*([A-Za-z][\w ]*?)|([A-Za-z][\w ]*?)\s+(hours?|kg|units?))(?=\s+and|,|\.|$)",
        text,
        re.I,
    ):
        resource = (resource_after_unit or resource_before_unit).strip()
        resource = re.sub(r"^(?:hours?|kg|units?)\s+", "", resource, flags=re.I)
        values[_clean_name(resource)] = float(value)
    return values

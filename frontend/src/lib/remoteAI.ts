/* Optional browser-direct AI extraction.
   If the site owner sets VITE_AI_BASE_URL / VITE_AI_API_KEY / VITE_AI_MODEL
   (e.g. a referrer-restricted Gemini key), free-form problems are sent to an
   OpenAI-compatible /chat/completions endpoint, the returned model JSON is
   validated, and the plan is solved locally in the browser.
   Nothing here runs unless a key is configured. */

import {
  detectCurrency,
  solveGameActions,
  solveProcurement,
  solveProduction,
  solveTransportation,
  type Analysis,
} from './localSolver';

interface AIConfig { baseUrl?: string; apiKey?: string; model?: string }

let override: AIConfig | null = null;

/** App entry-point calls this once with its (Vite) environment. Tests inject here too. */
export function configureAI(config: AIConfig): void {
  override = config;
}

function readConfig(): AIConfig {
  return {
    baseUrl: override?.baseUrl?.replace(/\/+$/, ''),
    apiKey: override?.apiKey,
    model: override?.model,
  };
}

export function isAIConfigured(): boolean {
  const c = readConfig();
  return Boolean(c.baseUrl && c.apiKey && c.model);
}

const PROMPT = `You are an operations-research model classifier and mathematical formulation assistant. Read the business problem and return JSON only.
Users phrase problems freely in their own words — never require rigid templates like "Product A". They may write casually ("I run a bakery…", "I ship from…", "my factory has…").
Understand synonyms: profit = profit/margin/earn/yield/contribution/gain/revenue/makes/gives/price; needs = needs/uses/requires/consumes/takes/spends; have = have/available/in stock/on hand/capacity/total/limited to/left.
Product names can be anything (chairs, bread, phones, X1). Resource names can be anything (wood, machine hours, labour, oven).
Places can be anything (Warehouse A, Mumbai depot, Store X). Costs can appear as "A to X costs 4", "A->X 4", "$4 from A to X", "cost from A to X is 4", "shipping A-X: 4".
Normalize names into the model, keep every stated number exact, never invent missing numbers.
Trace the problem first: identify whether it is production (make things for profit with limited resources), transportation (ship from supplies to demands with route costs), or game theory (two players, payoffs).
This site only solves optimization problems (production, transportation, procurement/shopping, game theory) — never general math.

Choose model_type as production, transportation, procurement, or game_theory. Return this envelope:
{"model_type":"production|transportation|procurement|game_theory","problem_summary":string,"technique":string,"objective":string,"variables":[string],"constraints":[string],"assumptions":[string],"model":{...}}

For production, model must be {"products":[{"name":string,"profit":number,"resource_usage":{resource:number},"demand_limit":number|null}],"capacities":{resource:number}}.
For transportation, model must be {"sources":[{"name":string,"supply":number}],"destinations":[{"name":string,"demand":number}],"costs":{"source":{"destination":number}}}.
For procurement (groceries, diet, vendor choice: meet minimum needs cheapest across shops), model must be {"shops":[string],"items":[string],"requirements":{item:number},"prices":{"shop":{item:number}},"discounts":[{"shop":string,"amount":number,"threshold":number}]}. List items in the order the requirements appear; map "respectively" prices to that order; keep every price exact.
For game_theory, model must be {"player_one_actions":[string],"player_two_actions":[string],"payoffs":{"player_one_action":{"player_two_action":number}},"zero_sum":true}. Use Player 1's payoff values and preserve negative payoffs.
Use consistent names, preserve all stated numbers, and do not invent missing numbers. If the problem is general math or otherwise neither supported type, return a concise JSON error with {"error":"unsupported"}.

User text:
`;

function stripCodeFence(content: string): string {
  const m = /\{[\s\S]*\}/.exec(content);
  return m ? m[0] : content;
}

interface AIModel {
  model_type: 'production' | 'transportation' | 'procurement' | 'game_theory';
  model: Record<string, unknown>;
  problem_summary?: string;
  technique?: string;
  objective?: string;
  variables?: string[];
  constraints?: string[];
  assumptions?: string[];
}

function num(v: unknown): number {
  const n = typeof v === 'string' ? parseFloat(v) : (v as number);
  if (!Number.isFinite(n)) throw new Error('bad-number');
  return n;
}

/** Validate the AI payload and solve it with the local engine. Throws on any problem. */
export async function extractWithAI(problem: string): Promise<Analysis> {
  const { baseUrl, apiKey, model } = readConfig();
  if (!baseUrl || !apiKey || !model) throw new Error('AI is not configured.');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 45000);
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
  try {
    // Retry once on transient failures (overloaded provider / flaky network).
    let response: Response | null = null;
    let networkError: unknown = null;
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        response = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            model,
            messages: [{ role: 'user', content: PROMPT + problem }],
            temperature: 0,
          }),
          signal: controller.signal,
        });
        networkError = null;
      } catch (error) {
        networkError = error;
        response = null;
        if (attempt < 2) await sleep(1200);
        continue;
      }
      if (response.ok) break;
      if (response.status === 401 || response.status === 403) {
        throw new Error('The site AI key was rejected. The owner needs to check the key and its restrictions.');
      }
      if (response.status === 429) throw new Error('The free AI quota is used up right now. Try an example or come back later.');
      if ((response.status === 502 || response.status === 503 || response.status === 504) && attempt < 2) {
        await sleep(1500);
        continue;
      }
      break;
    }
    if (!response) {
      if (networkError instanceof DOMException && networkError.name === 'AbortError') {
        throw new Error('The AI service took too long to answer. The optimal answer below was solved in your browser instead.');
      }
      throw new Error('Could not reach the AI service (network error or wrong base URL). The owner should check VITE_AI_BASE_URL.');
    }
    if (!response.ok) {
      if (response.status === 503 || response.status === 502 || response.status === 504) {
        throw new Error('The AI service is temporarily overloaded. The optimal answer below was solved in your browser instead.');
      }
      throw new Error(`AI request failed (${response.status}). The optimal answer below was solved in your browser instead.`);
    }
    const payload = (await response.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const content = payload.choices?.[0]?.message?.content ?? '';
    let parsed: AIModel & { error?: string };
    try {
      parsed = JSON.parse(stripCodeFence(content)) as AIModel & { error?: string };
    } catch {
      throw new Error('The AI returned an unusable model. Try rephrasing with exact numbers.');
    }
    if (parsed.error === 'unsupported') {
      throw new Error('This business problem is not yet supported. Supported models are production planning, transportation, and game theory.');
    }
    const meta = {
      problem_summary: parsed.problem_summary ?? '',
      technique: parsed.technique ?? '',
      objective: parsed.objective ?? '',
      variables: parsed.variables ?? [],
      constraints: parsed.constraints ?? [],
      assumptions: parsed.assumptions ?? [],
      currency: detectCurrency(problem),
    };
    if (parsed.model_type === 'production') {
      const m = parsed.model as {
        products?: { name?: unknown; profit?: unknown; resource_usage?: Record<string, unknown>; demand_limit?: unknown }[];
        capacities?: Record<string, unknown>;
      };
      if (!m.products?.length || !m.capacities) throw new Error('bad-model');
      const result = solveProduction(
        m.products.map((p) => ({
          name: String(p.name ?? 'Product'),
          profit: num(p.profit),
          resource_usage: Object.fromEntries(Object.entries(p.resource_usage ?? {}).map(([k, v]) => [k, num(v)])),
          demand_limit: p.demand_limit == null ? null : num(p.demand_limit),
        })),
        Object.fromEntries(Object.entries(m.capacities).map(([k, v]) => [k, num(v)])),
      );
      return { ...result, ...meta, model_type: 'production' };
    }
    if (parsed.model_type === 'transportation') {
      const m = parsed.model as {
        sources?: { name?: unknown; supply?: unknown }[];
        destinations?: { name?: unknown; demand?: unknown }[];
        costs?: Record<string, Record<string, unknown>>;
      };
      if (!m.sources?.length || !m.destinations?.length || !m.costs) throw new Error('bad-model');
      const result = solveTransportation({
        sources: m.sources.map((s) => ({ name: String(s.name), supply: num(s.supply) })),
        destinations: m.destinations.map((d) => ({ name: String(d.name), demand: num(d.demand) })),
        costs: Object.fromEntries(
          Object.entries(m.costs).map(([s, row]) => [s, Object.fromEntries(Object.entries(row).map(([d, v]) => [d, num(v)]))]),
        ),
      });
      return { ...result, ...meta, model_type: 'transportation' };
    }
    if (parsed.model_type === 'procurement') {
      const m = parsed.model as {
        shops?: unknown[];
        items?: unknown[];
        requirements?: Record<string, unknown>;
        prices?: Record<string, Record<string, unknown>>;
        discounts?: { shop?: unknown; amount?: unknown; threshold?: unknown }[];
      };
      const shops = (m.shops ?? []).map(String);
      const items = (m.items ?? []).map(String);
      if (shops.length === 0 || items.length === 0 || !m.requirements || !m.prices) throw new Error('bad-model');
      const requirements: Record<string, number> = {};
      for (const it of items) {
        const v = m.requirements[it];
        if (v == null) throw new Error('bad-model');
        requirements[it] = num(v);
      }
      const prices: Record<string, Record<string, number>> = {};
      for (const sh of shops) {
        const row = m.prices[sh];
        if (!row) throw new Error('bad-model');
        prices[sh] = {};
        for (const it of items) {
          const v = row[it];
          if (v != null) prices[sh][it] = num(v);
        }
        if (Object.keys(prices[sh]).length === 0) throw new Error('bad-model');
      }
      for (const it of items) {
        if (!shops.some((sh) => prices[sh][it] != null)) throw new Error('bad-model');
      }
      const result = solveProcurement(
        {
          shops,
          items,
          requirements,
          prices,
          discounts: (m.discounts ?? []).map((d) => ({ shop: String(d.shop), amount: num(d.amount), threshold: num(d.threshold) })),
        },
        detectCurrency(problem),
      );
      return { ...result, ...meta, model_type: 'procurement' as const };
    }
    if (parsed.model_type === 'game_theory') {
      const m = parsed.model as {
        player_one_actions?: unknown[];
        player_two_actions?: unknown[];
        payoffs?: Record<string, Record<string, unknown>>;
      };
      const p1 = (m.player_one_actions ?? []).map(String);
      const p2 = (m.player_two_actions ?? []).map(String);
      if (p1.length < 2 || p2.length < 2 || !m.payoffs) throw new Error('bad-model');
      const payoffs: Record<string, Record<string, number>> = {};
      for (const a of p1) {
        payoffs[a] = {};
        for (const b of p2) {
          const v = m.payoffs[a]?.[b];
          if (v == null) throw new Error('bad-model');
          payoffs[a][b] = num(v);
        }
      }
      const result = solveGameActions(p1[0], p1[1], payoffs);
      void p2;
      return { ...result, ...meta, model_type: 'game_theory' };
    }
    throw new Error('bad-model');
  } catch (error) {
    if (error instanceof Error && ['bad-model', 'bad-number'].includes(error.message)) {
      throw new Error('The AI returned an unusable model. Try rephrasing with exact numbers.');
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

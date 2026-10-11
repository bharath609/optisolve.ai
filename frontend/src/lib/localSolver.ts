/* Offline engine: understand free-form plain-English optimization problems
   (no API key) and solve them in-browser with the bundled simplex LP solver.
   Accepts any wording — product names, synonyms for profit/resources/capacity —
   then normalizes ("understands") it into a production / transportation /
   game-theory model. Mirrors backend/app/ai.py fallback parsing. */

import { solveLP } from './simplex';

export type ModelType = 'production' | 'transportation' | 'game_theory';

export interface ProductResult { name: string; quantity: number; profit: number }
export interface ResourceResult { name: string; used: number; capacity: number; utilization: number }
export interface ShipmentResult { source: string; destination: string; quantity: number; cost: number }
export interface StrategyResult { player: string; action: string; probability: number }

export interface Analysis {
  model_type: ModelType;
  problem_summary: string;
  technique: string;
  objective: string;
  variables: string[];
  constraints: string[];
  assumptions: string[];
  products: ProductResult[];
  resources: ResourceResult[];
  shipments: ShipmentResult[];
  strategies: StrategyResult[];
  game_value?: number;
  total_profit?: number;
  total_cost?: number;
  explanation: string;
}

/* ---------------- user-friendly clarification -------------------------------
   Instead of a dead-end error ("go away and add X"), the solver traces what
   it understood and exactly what is still missing. ClarificationError extends
   Error so old catch-blocks keep working, but the UI can render a helpful
   understanding card with one-click fixes. */

export type DetectedType = ModelType | 'production?' | 'transportation?' | 'game?' | 'unknown';

export class ClarificationError extends Error {
  detected: DetectedType;
  understood: string[];
  missing: string[];
  fixExample: string;
  appendFix: string;
  constructor(opts: {
    message: string;
    detected: DetectedType;
    understood?: string[];
    missing?: string[];
    fixExample?: string;
    appendFix?: string;
  }) {
    super(opts.message);
    this.name = 'ClarificationError';
    this.detected = opts.detected;
    this.understood = opts.understood ?? [];
    this.missing = opts.missing ?? [];
    this.fixExample = opts.fixExample ?? '';
    this.appendFix = opts.appendFix ?? '';
  }
}

const round2 = (v: number) => Math.round(v * 100) / 100;
const round4 = (v: number) => Math.round(v * 10000) / 10000;
/** Python-style {:.4g} formatting for the game explanation. */
function fmtG(v: number): string {
  if (Object.is(v, -0)) v = 0;
  const s = (parseFloat(v.toPrecision(4)) as number).toString();
  return s.replace('e+0', 'e+').replace('e-0', 'e-');
}

/* ---------------- shared helpers ------------------------------------------ */

const STOP_WORDS = new Set([
  'the', 'a', 'an', 'each', 'every', 'per', 'of', 'for', 'with', 'and', 'or',
  'to', 'from', 'in', 'on', 'is', 'are', 'it', 'its', 'we', 'our', 'you',
  'that', 'this', 'which', 'using', 'uses', 'use', 'needs', 'need', 'requires',
  'require', 'consumes', 'consume', 'takes', 'take', 'profit', 'profits',
  'margin', 'earn', 'earns', 'gives', 'give', 'yields', 'makes', 'make',
  'units', 'unit', 'hours', 'hour', 'hrs', 'hr', 'kg', 'kgs', 'tons', 'ton',
]);

function cleanName(raw: string): string {
  let s = raw.trim().replace(/\s+/g, ' ');
  s = s.replace(/^(the|a|an|each|every|our|your|my)\s+/i, '').trim();
  if (!s) return s;
  // Title-case short names, keep longer names as written
  return s;
}

function canonKey(s: string): string {
  const norm = s.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();
  const stripped = norm
    .replace(/\b(units?|hours?|hrs?|hr|kg|kgs|tons?|tonnes?|days?|minutes?|mins?)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return stripped || norm;
}

function escRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/* ---------------- production parsing (free-form) ---------------------------- */

function parseResourceValues(text: string): Record<string, number> {
  // Legacy strict pattern: requires explicit units (kept for compatibility).
  const values: Record<string, number> = {};
  const re =
    /(\d+(?:\.\d+)?)\s*(?:(hours?|kg|units?)\s+(?:on|of)?\s*([A-Za-z][\w ]*?)|([A-Za-z][\w ]*?)\s+(hours?|kg|units?))(?=\s+and|,|\.|$)/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const value = m[1];
    const unitFirst = m[2];
    let resource = (m[4] || m[3] || '').trim();
    if (!unitFirst) resource = resource.replace(/^(?:hours?|kg|units?)\s+/i, '');
    resource = resource.trim();
    if (resource) values[cleanName(resource)] = parseFloat(value);
  }
  return values;
}

/** Free-form "2 wood", "2 hours of machine time", "2 machine hours" etc. */
function parseUsageFreeForm(segment: string, excludeValue: number | null): Record<string, number> {
  const usage: Record<string, number> = {};
  // (?<![A-Za-z0-9_.]) so digits inside words like X1 are never read as quantities
  const re =
    /(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|kg|kgs|tons?|tonnes?|units?|days?|minutes?|mins?)?\s*(?:of\s+)?([A-Za-z][A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|\.|$|\+|needs?|uses?|requires?|consumes?|takes?|takes|per))/gi;
  let m: RegExpExecArray | null;
  let skippedProfit = false;
  while ((m = re.exec(segment)) !== null) {
    const value = parseFloat(m[1]);
    let resource = (m[3] || '').trim().replace(/\s+per\s+unit.*$/i, '').trim();
    // strip leading unit words accidentally captured
    resource = resource.replace(/^(units?|hours?|hrs?|kg)\s+/i, '').trim();
    const low = resource.toLowerCase();
    if (!resource || resource.length > 40) continue;
    if (/^(per|each|every)\b/i.test(resource)) continue;
    if (low.includes('profit') || low.includes('margin') || low.includes('earn') || low.includes('revenue')) continue;
    if (STOP_WORDS.has(low)) continue;
    // Skip the profit number itself once (e.g. "40 profit and needs 2 machine...")
    if (excludeValue != null && !skippedProfit && Math.abs(value - excludeValue) < 1e-9) {
      skippedProfit = true;
      continue;
    }
    const name = cleanName(resource);
    if (name) usage[name] = value;
  }
  return usage;
}

function productNameFromSentence(sentence: string, fallback: string): string {
  let m: RegExpExecArray | null;
  // "Product A" / "Product Bread"
  m = /(product\s+[A-Za-z0-9\-]+)/i.exec(sentence);
  if (m) return cleanName(m[1]);
  // "Profit per X1 is 40" — exact single-token name (avoids lazy-match traps like "X")
  m = /(?:profit|margin|earning|contribution|gain)\s+(?:per|for|of|on)\s+(?:of\s+)?([A-Za-z][A-Za-z0-9\-]*)/i.exec(sentence);
  if (m && !STOP_WORDS.has(m[1].trim().toLowerCase())) return cleanName(m[1]);
  // "Mango Smoothie for a profit of 50" — up to 3 words before "for a profit"
  m = /([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,2})\s+for\s+a\s+(?:profit|margin)\s+of\s+\$?\d/i.exec(sentence);
  if (m) {
    const cand = cleanName(m[1].replace(/^(sells?|makes?|produces?|offers?|manufactures?|provides?)\s+/i, ''));
    const last = (cand.toLowerCase().split(' ').pop() || '');
    if (cand && cand.length <= 30 && !STOP_WORDS.has(last)) return cand;
  }
  // "Chairs give 40", "Bread earns 25", "Phone yields 120"
  m = /([A-Za-z][A-Za-z0-9\- ]{0,28}?)\s+(?:gives?|earns?|yields?|makes?|brings?|fetches?|nets?)\s+(?:a\s+)?(?:profit\s+(?:of\s+)?)?\$?\d/i.exec(sentence);
  if (m && !STOP_WORDS.has(m[1].trim().toLowerCase())) {
    let cand = cleanName(m[1]);
    cand = cand.replace(/^(we|they|you|i)\s+/i, '').trim();
    cand = cand.replace(/^(make|makes|produces?|manufactures?|sells?|offers?|builds?)\s+/i, '').trim();
    cand = cand.replace(/\s+(that|which|who)$/i, '').trim();
    if (cand && !STOP_WORDS.has(cand.toLowerCase())) return cand;
  }
  // "profit for chairs is 40", "profit of tables: 30" (separator required so names like X1 stay intact)
  m = /(?:profit|margin|earning|contribution|gain)(?:\s*(?:per|for|of|on))?\s+(?:of\s+)?([A-Za-z][A-Za-z0-9\- ]{0,28}?)\s+(?:is|=|:|of)\s*\$?\d/i.exec(sentence);
  if (m && !STOP_WORDS.has(m[1].trim().toLowerCase())) {
    const cand = cleanName(m[1].replace(/\s+(is|are|of|at)$/i, ''));
    if (cand && cand.length <= 30) return cand;
  }
  // "Chairs: profit 40" / "Chairs - 40 profit"
  m = /([A-Za-z][A-Za-z0-9\- ]{0,28}?)\s*[:\-–]\s*\$?\d+(?:\.\d+)?\s*(?:profit|margin)/i.exec(sentence);
  if (m) return cleanName(m[1]);
  // "bread margin 25" / "chairs profit 40" — name before the keyword
  m = /([A-Za-z][A-Za-z0-9\-]*)\s+(?:profit|margin|earning|contribution)s?\s*(?:is|=|:|of)?\s*\$?\d+(?:\.\d+)?/i.exec(sentence);
  if (m && !STOP_WORDS.has(m[1].trim().toLowerCase())) return cleanName(m[1]);
  // "Each chair ..." / "One table ..."
  m = /(?:each|every|one|per)\s+([A-Za-z][A-Za-z0-9\-]*)/i.exec(sentence);
  if (m) {
    const w = m[1].trim();
    if (!STOP_WORDS.has(w.toLowerCase())) return cleanName(w.replace(/s$/, (s) => (s.length > 4 ? '' : s)) || w);
  }
  return fallback;
}

function profitFromSentence(sentence: string): number | null {
  let m: RegExpExecArray | null;
  // "Profit per X1 is 40", "margin for chairs: 30" (separator required so X1 isn't read as X + 1)
  m = /(?:profit|margin|earning|contribution|gain)\s+(?:per|for|of|on)\s+[A-Za-z0-9\- ]+?\s+(?:is|=|:)\s*\$?\s*(\d+(?:\.\d+)?)/i.exec(sentence);
  if (m) return parseFloat(m[1]);
  // number adjacent to profit synonyms: "40 profit", "profit of 40", "profit is $40", "earns 30"
  m = /(\d+(?:\.\d+)?)\s*(?:profit|margin|earned|earnings|contribution)/i.exec(sentence);
  if (m) return parseFloat(m[1]);
  m = /(?:profit|margin|earning|contribution|gain|revenue|nets?|earns?|yields?|makes?|gives?|price|value)(?:\s*(?:per|for|of|on|is|:|=))?\s*\$?\s*(\d+(?:\.\d+)?)/i.exec(sentence);
  if (m) return parseFloat(m[1]);
  return null;
}

function parseCapacitiesFree(text: string): Record<string, number> {
  const caps: Record<string, number> = {};
  const sentences = text.replace(/\$/g, '').split(/[.\n;]+/);
  const capHint = /have|available|in stock|on hand|capacity|capacities|at most|up to|limited|limit|maximum|total|only|supply\b/i;
  for (const raw of sentences) {
    const s = raw.trim();
    if (!s) continue;
    // Skip product-definition sentences unless they also state availability
    const isProductLine = /profit|margin|earn|contribution/i.test(s);
    if (isProductLine && !capHint.test(s)) continue;
    if (!capHint.test(s) && !/:/.test(s) && !/<=?/.test(s)) continue;
    // "Machine <= 100", "Machine: 100", "Machine capacity 100", "capacity of Machine is 100"
    let m: RegExpExecArray | null;
    const capWordRe = /([A-Za-z][A-Za-z ]{1,30}?)\s+(?:capacity|available|in stock|on hand)\s*(?:of\s*|is\s*|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:units?|hours?|hrs?|kg|tons?)?/gi;
    while ((m = capWordRe.exec(s)) !== null) {
      const name = cleanName(m[1].replace(/^(we|they|factory|we have|available|total|capacity of|capacity)\s+/i, ''));
      if (!name || STOP_WORDS.has(name.toLowerCase()) || /profit|margin|demand|supply/i.test(name)) continue;
      if (name.length > 40) continue;
      if (!(name in caps)) caps[name] = parseFloat(m[2]);
    }
    const leRe = /([A-Za-z][A-Za-z ]{1,30}?)\s*(?:capacity|available|in stock|on hand)?\s*(?:<=|<|:|=|is|of)\s*(\d+(?:\.\d+)?)\s*(?:units?|hours?|hrs?|kg|tons?)?/gi;
    while ((m = leRe.exec(s)) !== null) {
      const name = cleanName(m[1].replace(/^(we|they|factory|we have|available|total|capacity of|capacity)\s+/i, ''));
      if (!name || STOP_WORDS.has(name.toLowerCase()) || /profit|margin|demand|supply/i.test(name)) continue;
      if (name.length > 40) continue;
      caps[name] = parseFloat(m[2]);
    }
    // "100 units Machine", "100 hours of labor", "100 wood"
    const qtyRe = /(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*(units?|hours?|hrs?|hr|kg|kgs|tons?|tonnes?|days?)?\s*(?:of\s+)?([A-Za-z][A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|$|available|in stock|capacity|at most|maximum|total))/gi;
    while ((m = qtyRe.exec(s)) !== null) {
      const low3 = (m[3] || '').toLowerCase();
      if (/profit|margin|demand|supply|product/i.test(low3)) continue;
      const name = cleanName(m[3]);
      if (!name || STOP_WORDS.has(name.toLowerCase()) || name.length > 40) continue;
      if (!(name in caps)) caps[name] = parseFloat(m[1]);
    }
    // legacy strict units pattern as extra source
    for (const [k, v] of Object.entries(parseResourceValues(s))) {
      if (!(k in caps) && !/profit|demand|supply/i.test(k)) caps[k] = v;
    }
  }
  // "We have 100 units Machine and 80 units Labor" legacy anchor
  const capMatch = /we have (.*?)(?:demand|$)/i.exec(text.replace(/\$/g, ''));
  if (capMatch) {
    for (const [k, v] of Object.entries(parseResourceValues(capMatch[1]))) {
      if (!(k in caps)) caps[k] = v;
    }
  }
  return caps;
}

function parseDemandFree(text: string, productLabel: string): number | null {
  const label = productLabel.replace(/^product\s+/i, '').trim();
  const lastWord = (label.split(/\s+/).pop() || label).trim();
  const base = [productLabel, label, lastWord].filter(Boolean);
  // plural/singular tolerant variants: "cake" also matches "cakes"
  const cands: string[] = [];
  for (const b of base) {
    cands.push(b);
    if (/s$/i.test(b) && b.length > 3) cands.push(b.slice(0, -1));
    else cands.push(b + 's');
  }
  for (const cand of cands) {
    const e = escRe(cand);
    let m =
      new RegExp(`demand(?:\\s+for)?\\s+${e}\\s*(?:is\\s+)?(?:at most|max(?:imum)?|cannot exceed|up to|of|is|:|=|<=?)\\s*(\\d+(?:\\.\\d+)?)`, 'i').exec(text) ||
      new RegExp(`${e}\\s*(?:demand)?\\s*(?:at most|max(?:imum)?|cannot exceed|up to|limit(?:ed to)?|<=|<)\\s*(\\d+(?:\\.\\d+)?)`, 'i').exec(text) ||
      new RegExp(`(?:at most|max(?:imum)?|up to|no more than)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:units?\\s+)?(?:of\\s+)?${e}`, 'i').exec(text);
    if (m) return parseFloat(m[1]);
  }
  return null;
}

export interface ParsedProduct { name: string; profit: number; resource_usage: Record<string, number>; demand_limit: number | null }

/** Split "Profit per X1 is 40, profit per X2 is 30" into per-product clauses. */
function splitProductClauses(sentence: string): string[] {
  const parts = sentence.split(/,(?=\s*(?:and\s+)?(?:profit|margin|each|every|product\b))|\s+and\s+(?=[A-Za-z][A-Za-z0-9\- ]{0,30}?\s+for\s+a\s+(?:profit|margin))/i);
  if (parts.length <= 1) return [sentence];
  // only split if more than one part mentions a profit number
  const withProfit = parts.filter((p) => profitFromSentence(p) != null);
  if (withProfit.length > 1) return parts.map((p) => p.trim()).filter(Boolean);
  return [sentence];
}

/** Canonical-name match with safe substring fallback (used by attach + reconcile). */
function canonHit(a: string, b: string): boolean {
  const ka = canonKey(a);
  const kb = canonKey(b);
  if (!ka || !kb) return ka === kb && ka !== '';
  if (ka === kb) return true;
  // substring matching only for meaningful fragments (avoids "" or 1-char false hits)
  if (kb.length >= 3 && ka.includes(kb)) return true;
  if (ka.length >= 3 && kb.includes(ka)) return true;
  return false;
}

/**
 * Attach "X1 requires 2 machine hours..." lines (usage without profit) to
 * products discovered from profit-only clauses, matched by product name.
 */
function attachSplitNeeds(clean: string, products: ParsedProduct[]): void {
  const needy = products.filter((p) => Object.keys(p.resource_usage).length === 0);
  if (needy.length === 0) return;
  for (const raw of clean.split(/[.\n;]+/)) {
    const sentence = raw.trim();
    if (!sentence || /profit|margin|earn|contribution|gain|revenue/i.test(sentence)) continue;
    const needMatch = /^\s*(?:for\s+)?([A-Za-z][A-Za-z0-9\-]*)\s+(?:needs?|uses?|requires?|consumes?|takes?|takes up|spends?)\s+(.*)/i.exec(sentence);
    if (!needMatch) continue;
    const [, nameRaw, usageText] = needMatch;
    const usage = parseUsageFreeForm(usageText, null);
    if (Object.keys(usage).length === 0) continue;
    const target = needy.find((p) => canonHit(p.name, cleanName(nameRaw)));
    if (target) {
      target.resource_usage = usage;
    } else if (needy.some((p) => Object.keys(p.resource_usage).length === 0)) {
      // positional fallback: first still-empty product takes it
      const first = needy.find((p) => Object.keys(p.resource_usage).length === 0);
      if (first && !Object.keys(first.resource_usage).length) {
        // only use positionally when the names look generic/placeholder-like
        if (/^product\s+[a-z]$/i.test(first.name)) first.resource_usage = usage;
      }
    }
  }
}

export function parseProduction(text: string): { products: ParsedProduct[]; capacities: Record<string, number> } | null {
  const clean = text.replace(/\$/g, '');
  // Fast path: legacy strict format (keeps old behavior/tests exact).
  const strictProducts: ParsedProduct[] = [];
  for (const sentence of clean.split('.')) {
    const entry =
      /(Product\s+[A-Za-z0-9]+).*?(\d+(?:\.\d+)?)\s+profit.*?(?:needs|uses|requires|consumes)\s+(.*)/i.exec(sentence);
    if (!entry) continue;
    const [, name, profit, resourceText] = entry;
    const usage = parseResourceValues(`${profit} ${resourceText}`);
    const lastWord = name.trim().split(/\s+/).pop() as string;
    const demandMatch = new RegExp(
      `Demand for ${lastWord.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}.*?(?:at most|maximum|max)\\s*(\\d+(?:\\.\\d+)?)`,
      'i',
    ).exec(clean);
    if (Object.keys(usage).length > 0) {
      strictProducts.push({
        name: name.trim(),
        profit: parseFloat(profit),
        resource_usage: usage,
        demand_limit: demandMatch ? parseFloat(demandMatch[1]) : null,
      });
    }
  }
  if (strictProducts.length > 0) {
    const capMatch = /We have (.*?)(?:Demand|$)/i.exec(clean);
    const capacities: Record<string, number> = capMatch ? parseResourceValues(capMatch[1]) : {};
    const freeCaps = parseCapacitiesFree(clean);
    for (const [k, v] of Object.entries(freeCaps)) {
      if (!(k in capacities)) capacities[k] = v;
    }
    for (const product of strictProducts) {
      for (const resource of Object.keys(product.resource_usage)) {
        if (!(resource in capacities)) {
          const match = Object.keys(capacities).find(
            (c) => canonKey(c) === canonKey(resource) || c.toLowerCase().includes(resource.toLowerCase()) || resource.toLowerCase().includes(c.toLowerCase()),
          );
          if (match) {
            capacities[resource] = capacities[match];
            delete capacities[match];
          }
        }
      }
    }
    if (Object.keys(capacities).length > 0) return { products: strictProducts, capacities };
  }

  // Free-form path: any wording with profits + resource needs + availability.
  // Handles both "Chairs earn 40 and need 2 wood..." (same sentence) and
  // "Profit per X1 is 40. X1 requires 2 machine hours..." (split sentences).
  const sentences = clean.split(/[.\n;]+/);
  const products: ParsedProduct[] = [];
  let autoIdx = 0;
  for (const raw of sentences) {
    const sentence = raw.trim();
    if (!sentence) continue;
    // Split multi-product sentences: "Profit per X1 is 40, profit per X2 is 30"
    const clauses = splitProductClauses(sentence);
    for (const clause of clauses) {
      if (!/profit|margin|earn|contribution|gain|revenue/i.test(clause)) continue;
      // Skip capacity-only lines ("We have 100 wood...")
      if (/^(we have|available|in stock|capacity|total)/i.test(clause) && !/needs?|uses?|requires?|consumes?|takes?|needs/i.test(clause) && /have|available|capacity|total/i.test(clause)) {
        // still may define a product inline; check profit adjacency
        if (!/(gives?|earns?|yields?|makes?)/i.test(clause)) continue;
      }
      const profit = profitFromSentence(clause);
      if (profit == null || !(profit >= 0)) continue;
      autoIdx += 1;
      const name = productNameFromSentence(clause, `Product ${String.fromCharCode(64 + autoIdx)}`);
      // usage = needs/uses/... clause if present, else whole clause
      const needMatch = /(?:needs?|uses?|requires?|consumes?|takes?|takes up|spends?)\s+(.*)/i.exec(clause);
      const usageText = needMatch ? needMatch[1] : clause;
      const usage = parseUsageFreeForm(usageText, profit);
      if (Object.keys(usage).length === 0) {
        // profit-only clause (usage lives in another sentence) — remember it for phase 2
        if (!products.some((p) => canonKey(p.name) === canonKey(name))) {
          products.push({ name, profit, resource_usage: {}, demand_limit: null });
        }
        continue;
      }
      // avoid duplicates of the same product name
      if (products.some((p) => canonKey(p.name) === canonKey(name))) continue;
      products.push({ name, profit, resource_usage: usage, demand_limit: null });
    }
  }
  // Phase 2: attach "X1 requires 2 machine hours..." lines to profit-only products.
  attachSplitNeeds(clean, products);
  // Drop profit-only entries that never found usage (incomplete info, not a product).
  for (let i = products.length - 1; i >= 0; i--) {
    if (Object.keys(products[i].resource_usage).length === 0) products.splice(i, 1);
  }
  if (products.length === 0) return null;
  const capacities = parseCapacitiesFree(clean);
  // reconcile usage names with capacity names (e.g. "machines" vs "Machine")
  const capKeys = Object.keys(capacities);
  for (const product of products) {
    for (const res of Object.keys(product.resource_usage)) {
      if (res in capacities) continue;
      const hit = capKeys.find((c) => canonHit(c, res));
      if (hit) {
        product.resource_usage[hit] = product.resource_usage[res];
        delete product.resource_usage[res];
      }
    }
  }
  // drop usage resources with no known capacity only if we still have coverage
  const known = new Set(Object.keys(capacities).map(canonKey));
  for (const product of products) {
    for (const res of Object.keys(product.resource_usage)) {
      if (!known.has(canonKey(res))) {
        const hit = Object.keys(capacities).find((c) => canonHit(c, res));
        if (hit && hit !== res) {
          product.resource_usage[hit] = (product.resource_usage[hit] ?? 0) + product.resource_usage[res];
          delete product.resource_usage[res];
        }
      }
    }
  }
  for (const p of products) p.demand_limit = parseDemandFree(clean, p.name);
  const usedResources = new Set<string>();
  for (const p of products) for (const r of Object.keys(p.resource_usage)) usedResources.add(canonKey(r));
  const filteredCaps: Record<string, number> = {};
  for (const [k, v] of Object.entries(capacities)) {
    if (usedResources.has(canonKey(k))) filteredCaps[k] = v;
  }
  const finalCaps = Object.keys(filteredCaps).length > 0 ? filteredCaps : capacities;
  if (products.length === 0 || Object.keys(finalCaps).length === 0) return null;
  return { products, capacities: finalCaps };
}

export function solveProduction(products: ParsedProduct[], capacities: Record<string, number>): Analysis {
  const resources = Object.keys(capacities);
  const n = products.length;
  const c = products.map((p) => p.profit);
  const constraints = resources.map((r) => ({
    coeffs: products.map((p) => p.resource_usage[r] ?? 0),
    rhs: capacities[r],
    sense: '<=' as const,
  }));
  products.forEach((p, i) => {
    if (p.demand_limit != null) {
      const coeffs = new Array(n).fill(0);
      coeffs[i] = 1;
      constraints.push({ coeffs, rhs: p.demand_limit, sense: '<=' });
    }
  });
  const lp = solveLP(c, constraints);
  if (lp.status === 'unbounded') throw new Error('The problem is unbounded — add resource capacities or demand limits.');
  if (lp.status !== 'optimal') throw new Error('The solver could not find an optimal plan for these numbers.');
  const qtys = lp.solution.map((v) => round4(Math.max(0, v)));
  const productResults: ProductResult[] = products.map((p, i) => ({
    name: p.name,
    quantity: qtys[i],
    profit: round2(p.profit * qtys[i]),
  }));
  const resourceResults: ResourceResult[] = resources.map((r) => {
    const used = products.reduce((s, p, i) => s + (p.resource_usage[r] ?? 0) * qtys[i], 0);
    const cap = capacities[r];
    return { name: r, used: round4(used), capacity: cap, utilization: round4(cap ? used / cap : 0) };
  });
  const totalProfit = round2(lp.objective);
  const limited = resourceResults.filter((r) => r.utilization >= 0.999).map((r) => r.name);
  const understood = products.map((p) => `${p.name} (profit ${p.profit})`).join(', ');
  const explanation =
    limited.length > 0
      ? `Understood as: make ${understood} with limited ${resources.join(', ')}. The plan produces the highest possible profit of $${totalProfit.toLocaleString(undefined, { minimumFractionDigits: 2 })}. The limiting resource is ${limited.join(', ')}.`
      : `Understood as: make ${understood} with limited ${resources.join(', ')}. The plan produces the highest possible profit of $${totalProfit.toLocaleString(undefined, { minimumFractionDigits: 2 })} within the stated limits.`;
  return {
    model_type: 'production',
    problem_summary: 'Choose production quantities to maximize total profit under resource limits.',
    technique: 'Linear programming',
    objective: 'Maximize total profit',
    variables: products.map((p) => `quantity of ${p.name}`),
    constraints: resources.map((r) => `resource usage must not exceed ${r} capacity`),
    assumptions: ['Understood your wording automatically; production quantities may be fractional unless you require whole units.'],
    products: productResults,
    resources: resourceResults,
    shipments: [],
    strategies: [],
    total_profit: totalProfit,
    explanation,
  };
}

/* ---------------- transportation parsing + solving -------------------------- */

export interface TransportData {
  sources: { name: string; supply: number }[];
  destinations: { name: string; demand: number }[];
  costs: Record<string, Record<string, number>>;
}

function norm(s: string): string {
  return s.trim().replace(/\s+/g, ' ').toLowerCase();
}

function cleanNodeName(raw: string): string {
  let s = raw.trim().replace(/\s+/g, ' ');
  s = s.replace(/^(the|each|every|our|and|both|from|to|at|in|of)\s+/i, '').trim();
  // strip trailing verbs/adjectives captured by lazy regexes ("Factory A can" -> "Factory A")
  s = s.replace(/\s+(can|has|have|with|holds?|stocks?|suppl\w*|supply|produces?|makes?|offers?|provides?|needs?|requires?|demands?|demand|orders?|wants?)$/i, '').trim();
  s = s.replace(/\s+(factory|warehouse|plant|supplier|depot|store|shop|city|region|destination|source)s?$/i, (m) => m);
  return s;
}

export function parseTransportation(text: string): TransportData | null {
  // Production texts ("Demand for A at most 40", "We have 100 units Machine")
  // must never be mistaken for shipping. If profit wording is present without
  // any shipping nouns, this is not transportation.
  if (
    /profit|margin|earn|contribution|gain|revenue/i.test(text) &&
    !/warehouse|factory|plant|supplier|depot|store|shop|ship|freight|transport|deliver|customer|retailer/i.test(text)
  ) {
    return null;
  }
  const sources: { name: string; supply: number }[] = [];
  const destinations: { name: string; demand: number }[] = [];
  const seen = new Set<string>();
  const pushUnique = (arr: { name: string; supply: number }[] | { name: string; demand: number }[], name: string, qty: string, kind: 's' | 'd') => {
    const cleaned = cleanNodeName(name);
    if (!cleaned) return;
    // never treat pronouns or bare verbs as places ("We have 100" is capacity, not a warehouse)
    if (/^(we|they|you|he|she|it|there|here|this|that|they have|we have)$/i.test(cleaned)) return;
    const key = kind + ':' + norm(cleaned);
    if (seen.has(key)) return;
    seen.add(key);
    (arr as { name: string; supply: number }[]).push({ name: cleaned, supply: parseFloat(qty) } as never);
  };
  let m: RegExpExecArray | null;
  // 1) explicit supply phrasing — broad, any wording:
  //    "Warehouse A has supply 100", "A supplies 150", "100 units at A",
  //    "A: 100 units", "supply at A is 100", "A can ship 100", "A has 100"
  const supPatterns = [
    /([A-Za-z][\w\- ]*?)\s+(?:has\s+supply|suppl\w*(?:ies)?|supply)\s+(?:of\s+)?(\d+(?:\.\d+)?)/gi,
    /(?:supply|stock|inventory|capacity|available)\s+(?:at|from|of)\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)/gi,
    /(factory|warehouse|plant|supplier|depot|origin)\s+([A-Za-z0-9\- ]*?)\s*(?:has|holds?|stocks?|with|can supply|produces?|makes?|offers?)?\s*(?:supply|stock|inventory|capacity|available|units?)?\s*(?:of\s*|is\s*|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:units?)?/gi,
    /(\d+(?:\.\d+)?)\s*units?\s+(?:at|from|in)\s+([A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|\.|$))/gi,
    /([A-Za-z][\w\- ]*?)\s*(?:can\s+supply|can\s+ship|can\s+send|ships?|sends?|provides?|offers?)\s+(?:up\s+to\s+|about\s+)?(\d+(?:\.\d+)?)\s*(?:units?)?/gi,
    /([A-Za-z][\w\- ]*?)\s*(?:has|holds?|stocks?|with|contains?)\s+(?:up\s+to\s+|about\s+|around\s+)?(\d+(?:\.\d+)?)\s*units?\s*(?:supply|stock|inventory|available|to\s+ship|to\s+send)?/gi,
    /([A-Za-z][\w\- ]*?)\s*[:=]\s*(\d+(?:\.\d+)?)\s*units?\s*(?:supply|stock|available)?(?=\s*(?:and|,|;|\.|$))/gi,
  ];
  for (const re of supPatterns) {
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      if (m.length === 4) {
        // patterns with (keyword, place, qty) or (qty, place)
        if (/^\d/.test(m[1] ?? '')) {
          // "100 units at A" -> groups are (qty, place)
          if (/^\s*\d/.test(m[0]) && /at|from|in/i.test(m[0])) {
            pushUnique(sources, m[2], m[1], 's');
            continue;
          }
        }
        const place = `${m[1]} ${m[2]}`.trim();
        pushUnique(sources, place, m[3], 's');
      } else {
        // guard: "X: 100" must look supply-ish nearby to avoid eating costs/profits
        if (/^[^:]*[:=]\s*\d/.test(m[0]) && !/suppl|supply|stock|inventory|has|holds?|ships?|sends?|provides?|warehouse|factory|plant|depot|units?\s*(?:at|from)?/i.test(m[0] + ' ' + text.slice(Math.max(0, (m.index ?? 0) - 40), (m.index ?? 0)))) {
          // still allow "Warehouse A: 100" style (place word carries the hint)
          if (!/warehouse|factory|plant|supplier|depot|store|supply|stock/i.test(m[1])) continue;
        }
        // guard: lazy captures like "I have a warehouse A" (from "...with 100 units")
        // are not places — the real place ("warehouse A") is caught by the keyword pattern
        if (/\b(have|has|with|and|or)\b/i.test(m[1]) && !/^(factory|warehouse|plant|supplier|depot|origin|store|shop|city|customer)\b/i.test(m[1])) continue;
        pushUnique(sources, m[1], m[2], 's');
      }
    }
  }
  // 2) explicit demand phrasing, any wording: "Store X needs 120",
  //    "demand at Y is 130", "Y wants 50", "120 units for X", "X: 120 needed"
  const demPatterns = [
    /([A-Za-z][\w\- ]*?)\s+(?:needs?\s+demand|requires?\s+demand|has\s+demand|demand)\s+(?:of\s+|is\s+)?(\d+(?:\.\d+)?)/gi,
    /(?:demand|need|needs|requirement)\s+(?:at|for|of|from)\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)/gi,
    /(store|shop|city|customer|retailer|destination|market|client)\s+([A-Za-z0-9\- ]*?)\s*(?:needs?|requires?|demands?|wants?|orders?|takes?)?\s*(?:demand|units?|need)?\s*(?:of\s*|is\s*|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:units?)?/gi,
    /([A-Za-z][\w\- ]*?)\s+(?:needs?|requires?|demands?|orders?|wants?|takes?|receives?)\s+(\d+(?:\.\d+)?)\s*(?:units?)?(?!\s*(?:profit|margin))/gi,
    /(\d+(?:\.\d+)?)\s*units?\s+(?:for|to|at)\s+([A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|\.|$))/gi,
  ];
  for (const re of demPatterns) {
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      // production demand caps ("Demand for A at most 40") are never shipping destinations
      if (/at most|maximum|\bmax\b|up to|no more than|cannot exceed|\blimit/i.test(m[0])) continue;
      if (m.length === 4) {
        const place = `${m[1]} ${m[2]}`.trim();
        // guard: skip production-style "Demand for X at most N" (no adjacent number after 'demand')
        if (/at most|maximum|^demand for/i.test(m[0]) && !/\d+\s*$/.test(m[0].trim())) continue;
        pushUnique(destinations, place, m[3], 'd');
      } else if (/^\d/.test((m[1] ?? '').trim()) && m.length === 3 && /for|to|at/i.test(m[0])) {
        // "120 units for X" -> groups are (qty, place)
        if (/profit|margin|machine|labo?r|oven|wood|chips/i.test(m[0])) continue;
        pushUnique(destinations, m[2], m[1], 'd');
      } else {
        if (/demand for/i.test(m[0]) && /at most|maximum|max/i.test(text)) continue;
        // guard: generic "X needs N" must not fire on "X needs demand N" (verb is 'demand')
        if (/\b(needs?|requires?|demands?|demand|orders?)\s*$/i.test(m[1])) continue;
        if (/\b(demand|supply)\b/i.test(m[1]) && /needs?\s+demand|demand\s+\d/i.test(m[0])) continue;
        pushUnique(destinations, m[1], m[2], 'd');
      }
    }
  }
  // Drop destination hits that are actually sources and vice versa (same normalized name)
  const srcNorms = new Set(sources.map((s) => norm(s.name)));
  const dstNorms = new Set(destinations.map((d) => norm(d.name)));
  void dstNorms;
  for (let i = destinations.length - 1; i >= 0; i--) {
    if (srcNorms.has(norm(destinations[i].name))) destinations.splice(i, 1);
  }
  if (sources.length === 0 || destinations.length === 0) return null;
  // 3) costs, any wording: "A to X costs 4", "A -> X = 4", "A-X: 4",
  //    "shipping from A to X is $4 per unit", "$4 from A to X", "cost from A to X is 4"
  const costs: Record<string, Record<string, number>> = {};
  const srcNames = sources.map((s) => s.name);
  const dstNames = destinations.map((d) => d.name);
  const matchName = (token: string, known: string[]): string | null => {
    const t = norm(token);
    const exact = known.find((k) => norm(k) === t);
    if (exact) return exact;
    const withPrefix = known.find((k) => norm(k).endsWith(' ' + t));
    if (withPrefix) return withPrefix;
    const singles = known.filter((k) => (norm(k).split(' ').pop() ?? '') === t);
    if (singles.length === 1) return singles[0];
    const contains = known.filter((k) => norm(k).includes(t) || t.includes(norm(k)));
    if (contains.length === 1) return contains[0];
    // fuzzy: strip leading words ("shipping from A" -> "A") and retry each suffix
    const words = t.split(' ').filter((w) => w && !/^(shipping|shipment|freight|transport|transportation|delivery|deliver|costs?|costing|from|the|unit|units|per)$/.test(w));
    for (let i = 0; i < words.length; i++) {
      const suffix = words.slice(i).join(' ');
      const e2 = known.find((k) => norm(k) === suffix);
      if (e2) return e2;
      const w2 = known.find((k) => norm(k).endsWith(' ' + suffix));
      if (w2) return w2;
      const s2 = known.filter((k) => (norm(k).split(' ').pop() ?? '') === suffix);
      if (s2.length === 1) return s2[0];
    }
    return null;
  };
  const costPatterns = [
    /([A-Za-z][\w\- ]*?)\s*(?:->|→|—|–|-)\s*([A-Za-z][\w\- ]*?)\s*(?:costs?|costing|price|rate|freight|charge)?\s*(?:is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)(?:\s*per unit)?/gi,
    /([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)\s*(?:costs?|costing|price|rate|freight|charge|fare)?\s*(?:is|:|=|of)?\s*\$?\s*(\d+(?:\.\d+)?)(?:\s*per unit)?/gi,
    /(?:cost|price|rate|freight|charge|fare)\s+(?:from|of|for)\s+([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)(?:\s*per unit)?/gi,
    /\$?\s*(\d+(?:\.\d+)?)\s*(?:per unit\s+)?from\s+([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)(?=\s*(?:and|,|;|\.|$))/gi,
    /from\s+([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)\s*(?:costs?|is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)(?:\s*per unit)?/gi,
  ];
  for (const re of costPatterns) {
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      // "$4 from A to X" / "$4 per unit from A to X": first group is the cost number
      if (/^\d+(?:\.\d+)?$/.test((m[1] ?? '').trim())) {
        const cost = parseFloat(m[1]);
        const s = matchName(m[2], srcNames);
        const d = matchName(m[3], dstNames);
        if (s && d && Number.isFinite(cost)) {
          costs[s] = costs[s] ?? {};
          if (!(d in costs[s])) costs[s][d] = cost;
        }
        continue;
      }
      const s = matchName(m[1], srcNames);
      const d = matchName(m[2], dstNames);
      if (s && d) {
        costs[s] = costs[s] ?? {};
        if (!(d in costs[s])) costs[s][d] = parseFloat(m[3]);
      }
    }
  }
  const needed = sources.length * destinations.length;
  const have = Object.values(costs).reduce((acc, row) => acc + Object.keys(row).length, 0);
  if (have < needed) return null;
  // fix mis-typed arrays from pushUnique overload
  const fixedSources = sources.map((s) => ({ name: s.name, supply: (s as unknown as { supply: number; demand?: number }).supply ?? (s as unknown as { demand: number }).demand }));
  const fixedDests = (destinations as unknown as { name: string; supply: number }[]).map((d) => ({
    name: d.name,
    demand: (d as unknown as { demand: number }).demand ?? d.supply,
  }));
  return { sources: fixedSources, destinations: fixedDests, costs };
}

/** Best-effort transport understanding even when the problem is incomplete.
    Always returns what was found (possibly empty) plus the missing route pairs. */
export interface TransportPartial {
  sources: { name: string; supply: number }[];
  destinations: { name: string; demand: number }[];
  costs: Record<string, Record<string, number>>;
  missingRoutes: { source: string; destination: string }[];
  costCount: number;
  costNeeded: number;
}

export function parseTransportationPartial(text: string): TransportPartial {
  // Reuse the full parser's extraction but tolerate missing pieces.
  const full = parseTransportation(text);
  if (full) {
    return {
      sources: full.sources,
      destinations: full.destinations,
      costs: full.costs,
      missingRoutes: [],
      costCount: full.sources.length * full.destinations.length,
      costNeeded: full.sources.length * full.destinations.length,
    };
  }
  // Partial: extract sources/destinations with a lighter touch. Temporarily
  // patch: call internal extraction by relaxing the cost requirement.
  // We re-run source/dest regexes via a synthetic transport attempt:
  const sources: { name: string; supply: number }[] = [];
  const destinations: { name: string; demand: number }[] = [];
  const seen = new Set<string>();
  const push = (arr: { name: string; supply: number }[] | { name: string; demand: number }[], name: string, qty: string, kind: 's' | 'd') => {
    const cleaned = cleanNodeName(name);
    if (!cleaned) return;
    if (/^(we|they|you|he|she|it|there|here|this|that|they have|we have)$/i.test(cleaned)) return;
    const key = kind + ':' + norm(cleaned);
    if (seen.has(key)) return;
    seen.add(key);
    const qtyNum = parseFloat(qty);
    if (kind === 's') (arr as { name: string; supply: number }[]).push({ name: cleaned, supply: qtyNum });
    else (arr as { name: string; demand: number }[]).push({ name: cleaned, demand: qtyNum });
  };
  let m: RegExpExecArray | null;
  const supRes = [
    /([A-Za-z][\w\- ]*?)\s+(?:has\s+supply|suppl\w*(?:ies)?|supply)\s+(?:of\s+)?(\d+(?:\.\d+)?)/gi,
    /(?:supply|stock|inventory|capacity|available)\s+(?:at|from|of)\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)/gi,
    /(factory|warehouse|plant|supplier|depot|origin)\s+([A-Za-z0-9\- ]*?)\s*(?:has|holds?|stocks?|with)?\s*(?:supply|stock|inventory|capacity|available|units?)?\s*(?:of\s*|is\s*|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:units?)?/gi,
    /([A-Za-z][\w\- ]*?)\s*(?:can\s+supply|can\s+ship|ships?|sends?)\s+(?:up\s+to\s+)?(\d+(?:\.\d+)?)/gi,
    /([A-Za-z][\w\- ]*?)\s*(?:has|holds?)\s+(?:up\s+to\s+)?(\d+(?:\.\d+)?)\s*units?/gi,
  ];
  for (const re of supRes) {
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      if (m.length === 4) {
        const place = `${m[1]} ${m[2]}`.trim();
        push(sources, place, m[3], 's');
        continue;
      }
      if (/\b(have|has|with|and|or)\b/i.test(m[1]) && !/^(factory|warehouse|plant|supplier|depot|origin|store|shop|city|customer)\b/i.test(m[1])) continue;
      push(sources, m[1], m[2], 's');
    }
  }
  const demRes = [
    /(?:demand|need|needs|requirement)\s+(?:at|for|of|from)\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)/gi,
    /([A-Za-z][\w\- ]*?)\s+(?:needs?|requires?|demands?|orders?|wants?)\s+(\d+(?:\.\d+)?)/gi,
  ];
  for (const re of demRes) {
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      if (/at most|maximum|\bmax\b|up to|no more than|cannot exceed|\blimit/i.test(m[0])) continue;
      if (/profit|margin|machine|labo?r|oven|wood|chips/i.test(m[0])) continue;
      push(destinations, m[1], m[2], 'd');
    }
  }
  const srcNorms = new Set(sources.map((s) => norm(s.name)));
  for (let i = destinations.length - 1; i >= 0; i--) {
    if (srcNorms.has(norm(destinations[i].name))) destinations.splice(i, 1);
  }
  // costs found (best effort name match by last word)
  const costs: Record<string, Record<string, number>> = {};
  const matchLoose = (token: string, known: string[]): string | null => {
    const t = norm(token);
    const exact = known.find((k) => norm(k) === t);
    if (exact) return exact;
    const last = known.filter((k) => (norm(k).split(' ').pop() ?? '') === t || (norm(k).split(' ').pop() ?? '') === t.split(' ').pop());
    if (last.length === 1) return last[0];
    return null;
  };
  const costRes = [
    /([A-Za-z][\w\- ]*?)\s*(?:->|to)\s*([A-Za-z][\w\- ]*?)\s*(?:costs?|is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)/gi,
    /(?:cost|price)\s+(?:from|of|for)\s+([A-Za-z][\w\- ]*?)\s+to\s+([A-Za-z][\w\- ]*?)\s*(?:is|:|=)?\s*\$?\s*(\d+(?:\.\d+)?)/gi,
  ];
  const srcNames = sources.map((s) => s.name);
  const dstNames = destinations.map((d) => d.name);
  for (const re of costRes) {
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      const s = matchLoose(m[1], srcNames);
      const d = matchLoose(m[2], dstNames);
      if (s && d) {
        costs[s] = costs[s] ?? {};
        if (!(d in costs[s])) costs[s][d] = parseFloat(m[3]);
      }
    }
  }
  const fixedSources = sources.map((s) => ({ name: s.name, supply: (s as unknown as { supply: number }).supply }));
  const fixedDests = destinations.map((d) => ({ name: d.name, demand: (d as unknown as { demand: number }).demand ?? (d as unknown as { supply: number }).supply }));
  const missingRoutes: { source: string; destination: string }[] = [];
  for (const s of fixedSources) {
    for (const d of fixedDests) {
      if (!costs[s.name]?.[d.name]) missingRoutes.push({ source: s.name, destination: d.name });
    }
  }
  const costCount = fixedSources.length * fixedDests.length - missingRoutes.length;
  return { sources: fixedSources, destinations: fixedDests, costs, missingRoutes, costCount, costNeeded: fixedSources.length * fixedDests.length };
}

/** Best-effort production understanding for incomplete prompts. */
export function parseProductionPartial(text: string): {
  products: { name: string; profit: number | null }[];
  capacities: Record<string, number>;
} {
  const clean = text.replace(/\$/g, '');
  const products: { name: string; profit: number | null }[] = [];
  const seen = new Set<string>();
  for (const raw of clean.split(/[.\n;]+/)) {
    const sentence = raw.trim();
    if (!sentence) continue;
    if (!/profit|margin|earn|contribution|gain|revenue|makes?|yields?|gives?|price/i.test(sentence)) continue;
    const profit = profitFromSentence(sentence);
    // keep entries even when the profit number itself is missing ("chairs earn good profit")
    const name = productNameFromSentence(sentence, '');
    if (!name) continue;
    const key = canonKey(name);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    products.push({ name, profit });
  }
  let capacities: Record<string, number> = {};
  try {
    capacities = parseCapacitiesFree(clean);
  } catch {
    capacities = {};
  }
  return { products, capacities };
}

/** One-line trace of what the solver thinks the user is asking. */
export function traceProblem(text: string): { detected: DetectedType; understood: string[]; missing: string[] } {
  const hasProfitWords = /profit|margin|earn|contribution|gain|revenue/i.test(text);
  const hasShipNouns = /warehouse|factory|plant|supplier|depot|store|shop|ship|freight|transport|deliver|customer|retailer/i.test(text);
  // Production-first so "Demand for A at most 40" never traces as shipping.
  if (hasProfitWords && !hasShipNouns) {
    const prodOnly = parseProduction(text);
    if (prodOnly) {
      return {
        detected: 'production',
        understood: [
          `products: ${prodOnly.products.map((p) => `${p.name} (profit ${p.profit})`).join(', ')}`,
          `limits: ${Object.entries(prodOnly.capacities).map(([k, v]) => `${k} (${v})`).join(', ')}`,
        ],
        missing: [],
      };
    }
    const pp = parseProductionPartial(text);
    const understood: string[] = [];
    const missing: string[] = [];
    if (pp.products.length > 0) understood.push(`products: ${pp.products.map((p) => (p.profit != null ? `${p.name} (profit ${p.profit})` : p.name)).join(', ')}`);
    else missing.push('products + profit each');
    if (Object.keys(pp.capacities).length > 0) understood.push(`limits: ${Object.entries(pp.capacities).map(([k, v]) => `${k} (${v})`).join(', ')}`);
    else missing.push('what limits you');
    return { detected: 'production?', understood, missing };
  }
  const t = parseTransportation(text);
  if (t) {
    return {
      detected: 'transportation',
      understood: [
        `sources: ${t.sources.map((s) => `${s.name} (${s.supply})`).join(', ')}`,
        `destinations: ${t.destinations.map((d) => `${d.name} (${d.demand})`).join(', ')}`,
        `costs: ${Object.values(t.costs).reduce((a, r) => a + Object.keys(r).length, 0)} routes`,
      ],
      missing: [],
    };
  }
  if (isTransportLike(text)) {
    const p = parseTransportationPartial(text);
    const understood: string[] = [];
    const missing: string[] = [];
    if (p.sources.length > 0) understood.push(`sources: ${p.sources.map((s) => `${s.name} (${s.supply})`).join(', ')}`);
    else missing.push('supply quantities (e.g. "Warehouse A has 100")');
    if (p.destinations.length > 0) understood.push(`destinations: ${p.destinations.map((d) => `${d.name} (${d.demand})`).join(', ')}`);
    else missing.push('demand quantities (e.g. "Store X needs 120")');
    if (p.costCount > 0) understood.push(`route costs found: ${p.costCount}/${p.costNeeded}`);
    if (p.missingRoutes.length > 0 && p.sources.length > 0 && p.destinations.length > 0) {
      missing.push(`route costs: ${p.missingRoutes.slice(0, 4).map((r) => `${r.source} to ${r.destination}`).join(', ')}${p.missingRoutes.length > 4 ? ` (+${p.missingRoutes.length - 4} more)` : ''}`);
    } else if (p.costCount === 0 && p.sources.length > 0 && p.destinations.length > 0) {
      missing.push('every route cost (e.g. "A to X costs 4")');
    }
    return { detected: 'transportation?', understood, missing };
  }
  const prod = parseProduction(text);
  if (prod) {
    return {
      detected: 'production',
      understood: [
        `products: ${prod.products.map((p) => `${p.name} (profit ${p.profit})`).join(', ')}`,
        `limits: ${Object.entries(prod.capacities).map(([k, v]) => `${k} (${v})`).join(', ')}`,
      ],
      missing: [],
    };
  }
  if (/profit|margin|earn|contribution|makes?|produces?|manufactures?|capacity|available|machine|labo?r|wood|oven|material/i.test(text)) {
    const pp = parseProductionPartial(text);
    const understood: string[] = [];
    const missing: string[] = [];
    if (pp.products.length > 0) understood.push(`products: ${pp.products.map((p) => (p.profit != null ? `${p.name} (profit ${p.profit})` : p.name)).join(', ')}`);
    else missing.push('products + profit each (e.g. "chairs earn 40 each")');
    if (Object.keys(pp.capacities).length > 0) understood.push(`limits: ${Object.entries(pp.capacities).map(([k, v]) => `${k} (${v})`).join(', ')}`);
    else missing.push('what limits you (e.g. "we have 100 wood and 80 labor")');
    return { detected: 'production?', understood, missing };
  }
  if (GAME_WORDS.test(text)) return { detected: 'game?', understood: ['competitive / game-theory wording'], missing: ['two choices + four payoffs for Player 1'] };
  return { detected: 'unknown', understood: [], missing: ['what you make or ship, your limits, and your numbers'] };
}

export function solveTransportation(data: TransportData): Analysis {
  const { sources, destinations, costs } = data;
  const totalSupply = sources.reduce((s, x) => s + x.supply, 0);
  const totalDemand = destinations.reduce((s, x) => s + x.demand, 0);
  if (totalSupply + 1e-9 < totalDemand) {
    throw new Error(
      `Total supply (${totalSupply}) is less than total demand (${totalDemand}) — no feasible shipping plan exists.`,
    );
  }
  const nS = sources.length;
  const nD = destinations.length;
  const idx = (i: number, j: number) => i * nD + j;
  const c: number[] = [];
  for (let i = 0; i < nS; i++) for (let j = 0; j < nD; j++) c.push(-costs[sources[i].name][destinations[j].name]);
  const constraints: { coeffs: number[]; rhs: number; sense: '<=' | '>=' | '=' }[] = [];
  for (let i = 0; i < nS; i++) {
    const coeffs = new Array(nS * nD).fill(0);
    for (let j = 0; j < nD; j++) coeffs[idx(i, j)] = 1;
    constraints.push({ coeffs, rhs: sources[i].supply, sense: '<=' });
  }
  for (let j = 0; j < nD; j++) {
    const coeffs = new Array(nS * nD).fill(0);
    for (let i = 0; i < nS; i++) coeffs[idx(i, j)] = 1;
    constraints.push({ coeffs, rhs: destinations[j].demand, sense: '>=' });
  }
  const lp = solveLP(c, constraints);
  if (lp.status !== 'optimal') throw new Error('The solver could not find an optimal transportation plan for these numbers.');
  const shipments: ShipmentResult[] = [];
  for (let i = 0; i < nS; i++) {
    for (let j = 0; j < nD; j++) {
      const q = Math.max(0, lp.solution[idx(i, j)]);
      if (q > 0.0001) {
        shipments.push({
          source: sources[i].name,
          destination: destinations[j].name,
          quantity: round4(q),
          cost: round2(costs[sources[i].name][destinations[j].name] * q),
        });
      }
    }
  }
  const totalCost = round2(-lp.objective);
  return {
    model_type: 'transportation',
    problem_summary: 'Choose shipment quantities to minimize total shipping cost.',
    technique: 'Transportation linear programming',
    objective: 'Minimize total shipping cost',
    variables: sources.flatMap((s) => destinations.map((d) => `quantity from ${s.name} to ${d.name}`)),
    constraints: [
      ...sources.map((s) => `shipments from ${s.name} must not exceed supply ${s.supply}`),
      ...destinations.map((d) => `deliveries to ${d.name} must meet demand ${d.demand}`),
    ],
    assumptions: ['Understood your wording automatically; shipping cost is linear in quantity and fractional units are allowed.'],
    products: [],
    resources: [],
    shipments,
    strategies: [],
    total_cost: totalCost,
    explanation: `Understood as shipping from ${sources.map((s) => s.name).join(', ')} to ${destinations.map((d) => d.name).join(', ')}. The lowest-cost transportation plan has a total shipping cost of $${totalCost.toLocaleString(undefined, { minimumFractionDigits: 2 })}.`,
  };
}

/* ---------------- game theory parsing + solving ----------------------------- */

export function solveGameActions(
  first: string,
  second: string,
  payoffs: Record<string, Record<string, number>>,
): Analysis {
  const p2 = [first, second];
  // max v s.t. sum p = 1, sum_a payoff[a][r]*p[a] >= v; v free -> v+ - v-
  // vars: p[first], p[second], vPos, vNeg
  const c = [0, 0, 1, -1];
  const constraints: { coeffs: number[]; rhs: number; sense: '<=' | '>=' | '=' }[] = [
    { coeffs: [1, 1, 0, 0], rhs: 1, sense: '=' },
  ];
  for (const r of p2) {
    constraints.push({ coeffs: [payoffs[first][r], payoffs[second][r], -1, 1], rhs: 0, sense: '>=' });
  }
  const lp = solveLP(c, constraints);
  if (lp.status !== 'optimal') throw new Error('The solver could not find an optimal game strategy.');
  const probs = [Math.max(0, lp.solution[0]), Math.max(0, lp.solution[1])];
  const gameValue = round4(lp.solution[2] - lp.solution[3]);
  const strategies: StrategyResult[] = [];
  [first, second].forEach((a, i) => {
    if (probs[i] > 0.0001) strategies.push({ player: 'Player 1', action: a, probability: round4(probs[i]) });
  });
  return {
    model_type: 'game_theory',
    problem_summary: 'Choose a mixed strategy for Player 1 against a competing Player 2.',
    technique: 'Zero-sum game linear programming',
    objective: "Maximize Player 1's guaranteed expected payoff",
    variables: [`probability of ${first}`, `probability of ${second}`],
    constraints: [
      'strategy probabilities sum to 1',
      'expected payoff is at least the game value against every opposing action',
    ],
    assumptions: ['The game is zero-sum and both players choose actions simultaneously.'],
    products: [],
    resources: [],
    shipments: [],
    strategies,
    game_value: gameValue,
    explanation: `The optimal mixed strategy guarantees an expected payoff of ${fmtG(gameValue)} for Player 1.`,
  };
}

function detectActions(text: string): [string, string] | null {
  let m: RegExpExecArray | null;
  m = /choose\s+([A-Za-z]+)\s+or\s+([A-Za-z]+)/i.exec(text);
  if (m) return [m[1], m[2]];
  m = /(?:actions?|strategies?|options?|moves?|choices?)\s*(?:are|:)?\s*([A-Za-z]+)\s+(?:and|or|,)\s+([A-Za-z]+)/i.exec(text);
  if (m && /player|payoff|game|versus|vs|opponent|compet/i.test(text)) return [m[1], m[2]];
  m = /\b([A-Z][a-z]+)\s+(?:or|vs\.?|versus)\s+([A-Z][a-z]+)\b/.exec(text);
  if (m && /player|payoff|game/i.test(text)) return [m[1], m[2]];
  return null;
}

function parseGame(text: string): Analysis | null {
  if (!/player\s+1|player\s+2|payoff|expected payoff|zero-sum|game|opponent|versus|\bvs\b/i.test(text)) return null;
  const actions = detectActions(text);
  if (!actions) return null;
  const [first, second] = actions;
  const listed = /payoffs?\s+(?:are|of|is)?\s*\[?(-?\d+(?:\.\d+)?)\s*(?:,\s*and|,|and|;|\s)\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*and|,|and|;|\s)\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*and|,|and|;|\s)\s*(-?\d+(?:\.\d+)?)\]?/i.exec(text);
  if (listed) {
    const v = listed.slice(1, 5).map(Number);
    if (v.every((x) => Number.isFinite(x))) {
      return solveGameActions(first, second, {
        [first]: { [first]: v[0], [second]: v[1] },
        [second]: { [first]: v[2], [second]: v[3] },
      });
    }
  }
  // matrix form [[a,b],[c,d]]
  const matrixNums = text.match(/-?\d+(?:\.\d+)?/g)?.map(Number) ?? [];
  if (/\[\s*\[?-?\d[\s\S]*?,\s*-?\d[\s\S]*?\]/ .test(text) && matrixNums.length >= 4) {
    const v = matrixNums.slice(0, 4);
    return solveGameActions(first, second, {
      [first]: { [first]: v[0], [second]: v[1] },
      [second]: { [first]: v[2], [second]: v[3] },
    });
  }
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const patterns: [string, string, RegExp][] = [
    [first, first, new RegExp(`both\\s+${esc(first)}.*?(-?\\d+(?:\\.\\d+)?)`, 'i')],
    [first, second, new RegExp(`Player\\s+1\\s+${esc(first)}.*?Player\\s+2\\s+${esc(second)}.*?(-?\\d+(?:\\.\\d+)?)`, 'i')],
    [second, first, new RegExp(`Player\\s+1\\s+${esc(second)}.*?Player\\s+2\\s+${esc(first)}.*?(-?\\d+(?:\\.\\d+)?)`, 'i')],
    [second, second, new RegExp(`both\\s+${esc(second)}.*?(-?\\d+(?:\\.\\d+)?)`, 'i')],
  ];
  const payoffs: Record<string, Record<string, number>> = {};
  let found = 0;
  for (const [a, b, re] of patterns) {
    const mm = re.exec(text);
    if (mm) {
      payoffs[a] = payoffs[a] ?? {};
      payoffs[a][b] = parseFloat(mm[1]);
      found += 1;
    }
  }
  if (found === 4) return solveGameActions(first, second, payoffs);
  // last resort: any 4 numbers after the word "payoff"
  const after = /payoff[\s\S]*/i.exec(text)?.[0] ?? '';
  const nums = after.match(/-?\d+(?:\.\d+)?/g)?.map(Number) ?? [];
  if (nums.length >= 4) {
    const v = nums.slice(0, 4);
    return solveGameActions(first, second, {
      [first]: { [first]: v[0], [second]: v[1] },
      [second]: { [first]: v[2], [second]: v[3] },
    });
  }
  return null;
}

/* ---------------- dispatcher ------------------------------------------------ */

const TRANSPORT_WORDS = /warehouse|ship|shipping|transport|freight|supplier|customer|distribution|supply|demand|destination|source|deliver/i;
const GAME_WORDS = /player\s+[12]|payoff|zero-sum|mixed strateg|opponent|versus|\bvs\b/i;
const GENERAL_MATH = /solve\s+(for\s+)?x\b|derivative|integral|quadratic|differentiate|integrate|simplify|factor(ize|ise)?|eigen|matrix\s+inverse|determinant|limit\s+of|dy\/dx/i;

export function isTransportLike(text: string): boolean {
  return TRANSPORT_WORDS.test(text);
}

function looksLikeGeneralMath(text: string): boolean {
  if (!GENERAL_MATH.test(text)) return false;
  // optimization keywords take precedence — it's an OR problem, not algebra
  if (/profit|margin|capacity|supply|demand|shipping|payoff|resource|machine|labo?r|oven|chips|warehouse|store/i.test(text)) return false;
  return true;
}

/** Solve entirely in-browser. Throws Error with a user-friendly message on failure. */
export function solveLocally(problem: string): Analysis {
  const text = problem.trim();
  if (text.length < 10) throw new Error('Please describe your optimization problem in your own words — for example what you make, what limits you, or where you ship.');

  if (looksLikeGeneralMath(text)) {
    throw new Error(
      'OptiSolve only solves optimization problems — production planning, transportation, and game theory — not general math like equations or calculus. Describe profits with limited resources, shipping supplies/demands/costs, or a game payoff, in any wording.',
    );
  }

  const game = parseGame(text);
  if (game) return game;

  // Production-first: profit wording without shipping nouns is never transport
  // (stops "Demand for A at most 40" / "We have 100 units" misfiring as shipping).
  const hasProfitWords = /profit|margin|earn|contribution|gain|revenue/i.test(text);
  const hasShipNouns = /warehouse|factory|plant|supplier|depot|store|shop|ship|freight|transport|deliver|customer|retailer/i.test(text);
  if (hasProfitWords && !hasShipNouns) {
    const prodOnly = parseProduction(text);
    if (prodOnly) return solveProduction(prodOnly.products, prodOnly.capacities);
    // fall through to the friendly production clarification below
  } else {
    const transportWords = isTransportLike(text);
    const t = parseTransportation(text);
    if (t) return solveTransportation(t);
    if (transportWords) {
    // Shipping intent detected but something is missing — trace exactly what,
    // so the UI can show understanding + one-line fix instead of a dead end.
    const p = parseTransportationPartial(text);
    const fmtList = (items: string[]) => items.join(', ');
    if (p.sources.length > 0 && p.destinations.length > 0 && p.missingRoutes.length > 0) {
      const missingStr = p.missingRoutes.slice(0, 4).map((r) => `${r.source} to ${r.destination} costs ?`).join(', ');
      const first = p.missingRoutes[0];
      const fixBit = `${first.source} to ${first.destination} costs 5`;
      const understood = [
        `sources: ${fmtList(p.sources.map((s) => `${s.name} (${s.supply})`))}`,
        `destinations: ${fmtList(p.destinations.map((d) => `${d.name} (${d.demand})`))}`,
        `route costs so far: ${p.costCount}/${p.costNeeded}`,
      ];
      const missing = [`route costs: ${p.missingRoutes.map((r) => `${r.source} → ${r.destination}`).join(', ')}`];
      throw new ClarificationError({
        message:
          `Got it — this is a shipping problem and I understood most of it (${fmtList(understood)}). ` +
          `I'm still missing ${p.missingRoutes.length} route cost${p.missingRoutes.length > 1 ? 's' : ''}: ${missingStr}${p.missingRoutes.length > 4 ? '…' : ''}. ` +
          `Just add them in your own words — e.g. "${fixBit}" — and I'll solve it instantly.`,
        detected: 'transportation?',
        understood,
        missing,
        fixExample: `Shipping costs: ${p.missingRoutes.map((r) => `${r.source} to ${r.destination} costs 5`).join(', ')}.`,
        appendFix: ` Shipping costs: ${p.missingRoutes.map((r) => `${r.source} to ${r.destination} costs 5`).join(', ')}.`,
      });
    }
    if (p.sources.length > 0 || p.destinations.length > 0) {
      const understood: string[] = [];
      if (p.sources.length > 0) understood.push(`sources: ${fmtList(p.sources.map((s) => `${s.name} (${s.supply})`))}`);
      if (p.destinations.length > 0) understood.push(`destinations: ${fmtList(p.destinations.map((d) => `${d.name} (${d.demand})`))}`);
      if (p.costCount > 0) understood.push(`costs found: ${p.costCount}`);
      const missing: string[] = [];
      if (p.sources.length === 0) missing.push('who supplies how much (e.g. "Warehouse A has 100")');
      if (p.destinations.length === 0) missing.push('who needs how much (e.g. "Store X needs 120")');
      if (p.sources.length > 0 && p.destinations.length > 0) missing.push('every route cost (e.g. "A to X costs 4")');
      throw new ClarificationError({
        message:
          `Got it — this is a shipping problem${understood.length > 0 ? ` (${fmtList(understood)})` : ''}. ` +
          `To solve it I still need ${missing.join(' + ')} — in any wording you like. ` +
          `Example that always works: "Warehouse A has supply 100. Store X needs 120. A to X costs 4."`,
        detected: 'transportation?',
        understood,
        missing,
        fixExample: 'Warehouse A has supply 100. Warehouse B has supply 150. Store X needs 120. Store Y needs 130. Shipping costs: A to X costs 4, A to Y costs 6, B to X costs 5, B to Y costs 3.',
        appendFix: '',
      });
    }
    // Vague shipping intent with no numbers at all — guide, don't scold.
    throw new ClarificationError({
      message:
        `Got it — you want to optimize shipping. Tell me in your own words: (1) who supplies how much, (2) who needs how much, (3) what each route costs. ` +
        `Example: "Warehouse A has 100 units, Store X needs 120 units, shipping from A to X costs 4 per unit." Any wording with the same numbers works.`,
      detected: 'transportation?',
      understood: ['shipping intent'],
      missing: ['supplies', 'demands', 'route costs'],
      fixExample: 'Warehouse A has supply 100. Store X needs demand 120. Shipping costs: A to X costs 4.',
      appendFix: '',
    });
    }
  }

  const prod = parseProduction(text);
  if (prod) return solveProduction(prod.products, prod.capacities);
  // Production intent but incomplete — say exactly which piece is missing,
  // using the user's own product/resource names when we have them.
  if (/profit|margin|earn|contribution|gain|revenue|makes?|produces?|manufactures?|capacity|available|machine|labo?r|wood|oven|material|assembl/i.test(text)) {
    const pp = parseProductionPartial(text);
    if (pp.products.length > 0 || Object.keys(pp.capacities).length > 0) {
      const understood: string[] = [];
      if (pp.products.length > 0) {
        understood.push(`products: ${pp.products.map((p) => (p.profit != null ? `${p.name} (profit ${p.profit})` : `${p.name}`)).join(', ')}`);
      }
      if (Object.keys(pp.capacities).length > 0) {
        understood.push(`limits: ${Object.entries(pp.capacities).map(([k, v]) => `${k} (${v})`).join(', ')}`);
      }
      const missing: string[] = [];
      const needProfits = pp.products.some((p) => p.profit == null);
      if (pp.products.length === 0 || needProfits) missing.push('profit per item (e.g. "each chair earns 40")');
      const needUsage = true; // usage detail almost always the missing piece in partial parses
      if (Object.keys(pp.capacities).length === 0) missing.push('what limits you (e.g. "we have 100 wood and 80 labor")');
      else if (pp.products.length > 0) missing.push('what each item needs (e.g. "each chair needs 2 wood and 1 labor")');
      void needUsage;
      const firstProd = pp.products[0]?.name ?? 'chairs';
      // naive singular for a natural example ("chairs" -> "Each chair needs ...")
      const singular = /[^s]s$/i.test(firstProd) && firstProd.length > 3 ? firstProd.slice(0, -1) : firstProd;
      throw new ClarificationError({
        message:
          `Got it — this is a production problem${understood.length > 0 ? ` (${understood.join('; ')})` : ''}. ` +
          `To find your optimal plan I still need ${missing.join(' + ')}. ` +
          `Just continue in your own words — e.g. "Each ${singular} needs 2 wood and 1 labor. We have 100 wood and 80 labor." — and I'll solve it.`,
        detected: 'production?',
        understood,
        missing,
        fixExample: `Each ${singular} needs 2 wood and 1 labor. We have 100 wood and 80 labor available.`,
        appendFix: ` Each ${singular} needs 2 wood and 1 labor. We have 100 wood and 80 labor available.`,
      });
    }
  }
  if (GAME_WORDS.test(text)) {
    throw new ClarificationError({
      message:
        'This looks like a game-theory problem. Tell me the two choices (e.g. "choose Attack or Defend") and the four payoffs for Player 1, in any wording, and I will find the optimal mixed strategy.',
      detected: 'game?',
      understood: ['competitive / game-theory wording'],
      missing: ['two choices + four payoffs'],
      fixExample: 'Two players choose Attack or Defend. Payoffs are 3, -1, -2 and 2.',
      appendFix: '',
    });
  }
  // Truly vague / general — trace intent and invite any wording.
  throw new ClarificationError({
    message:
      'Tell me about your problem in your own words and I\'ll work out the rest — for example: "We make chairs (40 profit, 2 wood + 1 labor) and tables (30 profit, 1 wood + 2 labor). We have 100 wood and 80 labor." ' +
      'Or shipping: "Warehouse A has 100, Store X needs 120, A to X costs 4." I handle production, transportation, and game-theory problems (not general math).',
    detected: 'unknown',
    understood: [],
    missing: ['what you make or ship, your limits, and your numbers'],
    fixExample: 'We make chairs (40 profit, 2 wood + 1 labor) and tables (30 profit, 1 wood + 2 labor). We have 100 wood and 80 labor.',
    appendFix: '',
  });
}

import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import './App.css'
import { solveLocally, traceProblem, type Analysis } from './lib/localSolver'
import { configureAI, extractWithAI, isAIConfigured } from './lib/remoteAI'

/* ---------- tiny inline icon set (no emoji, no deps) ---------- */
function I({ d, size = 18, fill = false }: { d: string; size?: number; fill?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={fill ? 'currentColor' : 'none'}
      stroke="currentColor" strokeWidth={fill ? 0 : 1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  )
}
const LogoIcon = () => <I fill size={19} d="M12 2l2.7 7.3L22 12l-7.3 2.7L12 22l-2.7-7.3L2 12l7.3-2.7z" />
const FactoryIcon = () => <I d="M3 21V10l5 3v-3l5 3V7h4v14zM7 17h2v2H7zm5 0h2v2h-2zm5 0h2v2h-2z" />
const TruckIcon = () => <I d="M2 6h12v10H2zM14 10h4l4 4v2h-8zM6 19a1.6 1.6 0 100-3.2A1.6 1.6 0 006 19zm12 0a1.6 1.6 0 100-3.2A1.6 1.6 0 0018 19z" />
const GameIcon = () => <I d="M6 12h4m-2-2v4m7-3h.01M18 13h.01M17.3 5H6.7a4 4 0 00-4 3.6L2 14a2.5 2.5 0 004.4 1.6L8 14h8l1.6 1.6A2.5 2.5 0 0022 14l-.7-5.4a4 4 0 00-4-3.6z" />
const CartIcon = () => <I d="M3 4h2l2.4 12.2a1.5 1.5 0 001.5 1.3h8.9a1.5 1.5 0 001.5-1.2L21 8H6m9 12a1.4 1.4 0 100-2.8A1.4 1.4 0 0015 20zm-7 0a1.4 1.4 0 100-2.8A1.4 1.4 0 008 20z" />
const PenIcon = () => <I d="M12 20h9M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z" />
const ZapIcon = () => <I d="M13 2L4 14h6l-1 8 9-12h-6z" />
const CheckIcon = () => <I d="M20 6L9 17l-5-5" />
const ArrowIcon = () => <I d="M5 12h14m-6-6l6 6-6 6" />
const SparkIcon = () => <I size={13} d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />

/* ---------- animated number ---------- */
function CountUp({ value, decimals = 2, prefix = '', trim = false }: { value: number; decimals?: number; prefix?: string; trim?: boolean }) {
  const [display, setDisplay] = useState(0)
  const current = useRef(0)
  useEffect(() => {
    const from = current.current
    const start = performance.now()
    const dur = 1100
    let raf = 0
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / dur)
      const e = 1 - Math.pow(1 - p, 3)
      const v = from + (value - from) * e
      current.current = v
      setDisplay(v)
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value])
  let text = display.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
  if (trim) text = text.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')
  return <>{prefix}{text}</>
}

/* ---------- scroll reveal ---------- */
function useReveal(dep: unknown) {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll('.reveal:not(.visible)'))
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('visible')
            io.unobserve(e.target)
          }
        }
      },
      { threshold: 0.1 },
    )
    els.forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [dep])
}

configureAI({
  baseUrl: import.meta.env.VITE_AI_BASE_URL as string | undefined,
  apiKey: import.meta.env.VITE_AI_API_KEY as string | undefined,
  model: import.meta.env.VITE_AI_MODEL as string | undefined,
})

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const EXAMPLES = [
  {
    label: 'Furniture workshop',
    tag: 'Production · works instantly',
    icon: 'factory' as const,
    prompt:
      'Product A gives 40 profit and needs 2 units Machine and 1 units Labor. Product B gives 30 profit and needs 1 units Machine and 2 units Labor. We have 100 units Machine and 80 units Labor. Demand for A at most 40.',
  },
  {
    label: 'Bakery plan',
    tag: 'Production · works instantly',
    icon: 'factory' as const,
    prompt:
      'Product Bread gives 25 profit and needs 3 units Oven and 2 units Labor. Product Cake gives 45 profit and needs 4 units Oven and 3 units Labor. We have 120 units Oven and 90 units Labor. Demand for Cake at most 20.',
  },
  {
    label: 'Electronics line',
    tag: 'Production · works instantly',
    icon: 'factory' as const,
    prompt:
      'Product Phone gives 120 profit and needs 2 units Chips and 1 units Assembly. Product Tablet gives 100 profit and needs 1 units Chips and 2 units Assembly. We have 200 units Chips and 180 units Assembly.',
  },
  {
    label: 'Shipping plan',
    tag: 'Transportation · works instantly',
    icon: 'truck' as const,
    prompt:
      'Warehouse A has supply 100. Warehouse B has supply 150. Store X needs demand 120. Store Y needs demand 130. Shipping costs: A to X costs 4, A to Y costs 6, B to X costs 5, B to Y costs 3.',
  },
  {
    label: 'Free-form chairs',
    tag: 'Production · any wording',
    icon: 'pen' as const,
    prompt:
      'We make chairs and tables. Each chair earns 40 profit and takes 2 wood and 1 labor. Each table earns 30 profit and takes 1 wood and 2 labor. We have 100 wood and 80 labor available. Max 40 chairs can be sold.',
  },
  {
    label: 'Grocery shopping',
    tag: 'Purchase plan · any wording',
    icon: 'cart' as const,
    prompt:
      'A family needs at least 10 kg of rice, 5 liters of cooking oil, and 4 kg of pulses per month. Shop A sells rice at ₹50/kg, cooking oil at ₹120/liter, and pulses at ₹100/kg. Shop B sells the same items at ₹45/kg, ₹130/liter, and ₹90/kg, respectively. Shop A offers a ₹100 discount on the total purchase if the purchase value is at least ₹500. How should the family distribute its purchases between the two shops to minimize total cost?',
  },
]

function ExampleIcon({ kind }: { kind: 'factory' | 'truck' | 'pen' | 'game' | 'cart' }) {
  if (kind === 'truck') return <TruckIcon />
  if (kind === 'pen') return <PenIcon />
  if (kind === 'game') return <GameIcon />
  if (kind === 'cart') return <CartIcon />
  return <FactoryIcon />
}

function App() {
  const [problem, setProblem] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [message, setMessage] = useState('')
  const [clarify, setClarify] = useState<{
    detected: string
    understood: string[]
    missing: string[]
    fixExample: string
    appendFix: string
  } | null>(null)
  const [loading, setLoading] = useState(false)
  const [backend, setBackend] = useState<'checking' | 'online' | 'offline'>('checking')
  const [copied, setCopied] = useState(false)
  const [solvedVia, setSolvedVia] = useState<'api' | 'ai' | 'browser' | null>(null)
  const [aiNote, setAiNote] = useState<string | null>(null)

  // Live trace: show what the solver understands as the user types.
  const trace = useMemo(() => {
    if (problem.trim().length < 10) return null
    try {
      return traceProblem(problem)
    } catch {
      return null
    }
  }, [problem])

  useEffect(() => {
    let cancelled = false
    fetch(`${API_URL}/`, { method: 'GET' })
      .then((res) => {
        if (!cancelled) setBackend(res.ok ? 'online' : 'offline')
      })
      .catch(() => {
        if (!cancelled) setBackend('offline')
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function handleSolve() {
    if (!problem.trim()) {
      setMessage('Describe your optimization problem in your own words first — what you make, ship, or buy, your limits, and costs — or try one of the examples below. Note: this solver handles production, transportation, shopping/purchase, and game-theory optimization only, not general math.')
      return
    }
    if (problem.trim().length < 10) {
      setMessage('Please add a little more detail in any wording — e.g. profits, what each item needs, and what you have available.')
      return
    }

    setLoading(true)
    setMessage('')
    setClarify(null)
    setAnalysis(null)
    setSolvedVia(null)
    setAiNote(null)

    // 1) Try the hosted API first (it adds GPT-6 Astra understanding when configured).
    try {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 15000)
      try {
        const response = await fetch(`${API_URL}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ problem }),
          signal: controller.signal,
        })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || 'The model could not be solved.')
        setAnalysis(data)
        setSolvedVia('api')
        setLoading(false)
        return
      } finally {
        clearTimeout(timer)
      }
    } catch {
      // fall through to the built-in offline solver
    }

    // 2) Browser-direct AI (free-form questions) when the owner configured a key.
    if (isAIConfigured()) {
      try {
        const ai = await extractWithAI(problem)
        setAnalysis(ai)
        setSolvedVia('ai')
        setLoading(false)
        return
      } catch (error) {
        // Remember why, so the UI can show it — then fall through to offline.
        const reason = error instanceof Error ? error.message : 'AI request failed.'
        setAiNote(reason)
        console.warn('[optiforge] AI assist failed:', reason)
      }
    }

    // 3) Offline engine: parse + solve right in the browser. No server, no key.
    try {
      const local = solveLocally(problem)
      setAnalysis(local)
      setSolvedVia('browser')
    } catch (error) {
      const err = error as Error & {
        understood?: string[]
        missing?: string[]
        fixExample?: string
        appendFix?: string
        detected?: string
      }
      // Structured clarification (what we understood + exactly what's missing)
      // renders as a helpful card, not a dead-end error.
      if (err && (Array.isArray(err.understood) || Array.isArray(err.missing))) {
        setClarify({
          detected: String(err.detected ?? 'unknown'),
          understood: err.understood ?? [],
          missing: err.missing ?? [],
          fixExample: err.fixExample ?? '',
          appendFix: err.appendFix ?? '',
        })
        setMessage(err.message)
      } else {
        setMessage(error instanceof Error ? error.message : 'Could not solve this problem. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  function applyFix() {
    if (!clarify) return
    const fix = clarify.appendFix || (clarify.fixExample ? ' ' + clarify.fixExample : '')
    if (!fix.trim()) return
    const next = problem.trim().endsWith('.') ? `${problem.trim()} ${fix.trim()}` : `${problem.trim()}. ${fix.trim()}`
    setProblem(next)
    setClarify(null)
    setMessage('')
    // Re-solve automatically after a tick so the new text is in state.
    setTimeout(() => {
      document.getElementById('solve-btn')?.click()
    }, 50)
  }

  function useFullExample() {
    const ex = clarify?.fixExample || EXAMPLES[4].prompt
    setProblem(ex)
    setClarify(null)
    setMessage('')
    document.getElementById('solver')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  function useExample(prompt: string) {
    setProblem(prompt)
    setMessage('')
    setClarify(null)
    document.getElementById('solver')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  function copyPlan() {
    if (!analysis) return
    const cur = analysis.currency ?? '$'
    const lines = [
      analysis.problem_summary,
      '',
      analysis.explanation,
      '',
      ...analysis.products.map((p) => `${p.name}: ${p.quantity} units (${cur}${p.profit})`),
      ...analysis.resources.map((r) => `${r.name}: ${r.used}/${r.capacity} (${Math.round(r.utilization * 100)}%)`),
      ...analysis.shipments.map((s) => `${s.source} -> ${s.destination}: ${s.quantity} units (${cur}${s.cost})`),
      ...analysis.purchases.map((s) => `${s.shop} -> ${s.item}: ${s.quantity} units (${cur}${s.cost})`),
      ...analysis.strategies.map((s) => `${s.action}: ${Math.round(s.probability * 100)}%`),
    ]
    navigator.clipboard?.writeText(lines.join('\n')).then(
      () => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1800)
      },
      () => setMessage('Copy failed — select the results manually.'),
    )
  }

  const charCount = problem.length

  useReveal(`${loading}-${analysis ? analysis.explanation : 'none'}-${backend}`)

  return (
    <div className="page">
      <div className="aurora" aria-hidden>
        <i />
        <i />
        <i />
      </div>
      <div className="grid-overlay" aria-hidden />
      <header className="navbar">
        <div className="brand">
          <span className="brand-mark"><LogoIcon /></span>
          <span>
            optisolve.ai
            <small>Operations research, simplified</small>
          </span>
        </div>
        <nav className="nav-links">
          <a href="#solver">Solver</a>
          <a href="#how">How it works</a>
          <a href="#use-cases">Use cases</a>
          <a href="https://github.com/bharath609/optisolve" target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
        <div className="nav-cta">
          <span className={`status status-${backend}`}>
            <i /> {backend === 'checking' ? 'Checking API…' : backend === 'online' ? 'API online' : 'Offline solver ready'}
          </span>
          <a className="btn btn-small" href="#solver">
            Start optimizing
          </a>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy reveal">
            <p className="eyebrow">
              <span className="pulse" /> AI-POWERED OPERATIONS RESEARCH
            </p>
            <h1>
              Turn operational complexity into <span className="gradient">clear decisions.</span>
            </h1>
            <p className="hero-text">
              Describe your business problem in <strong>plain English — any wording works</strong>. OptiSolve identifies the right model —
              production, transportation, shopping, or game theory — and returns a <strong>mathematically optimal plan</strong> in seconds.
            </p>
            <div className="hero-actions">
              <a className="btn btn-primary" href="#solver">
                Solve my plan <span aria-hidden><ArrowIcon /></span>
              </a>
              <button className="btn btn-ghost" onClick={() => useExample(EXAMPLES[0].prompt)} type="button">
                <PenIcon /> Try a live example
              </button>
            </div>
            <div className="hero-stats">
              <div>
                <strong>4</strong>
                <span>model types supported</span>
              </div>
              <div>
                <strong>&lt;10s</strong>
                <span>typical solve time</span>
              </div>
              <div>
                <strong>100%</strong>
                <span>optimal, not heuristic</span>
              </div>
            </div>
          </div>
          <div className="hero-card-wrap reveal">
            <span className="float-chip fc-1"><SparkIcon /> optimal <b>mix found</b></span>
            <div className="hero-card" aria-hidden>
              <div className="hero-card-head">
                <span />
                <span />
                <span />
                <span className="live-dot"><i />LIVE SOLVE</span>
              </div>
              <div className="hero-card-body">
                <div className="hero-metric-label">
                  <span>TOTAL PROFIT</span>
                  <em>LP · OPTIMAL</em>
                </div>
                <strong><CountUp value={2200} decimals={2} prefix="$" /></strong>
                <span className="hero-delta">▲ +18.4% vs heuristic plan</span>
                <div className="hero-bar">
                  <div className="hero-bar-head">
                    <span><i style={{ background: '#3ef0b0' }} />Product A</span>
                    <span>40 units · $1,600</span>
                  </div>
                  <i className="track"><b style={{ '--w': '84%' } as CSSProperties} /></i>
                </div>
                <div className="hero-bar">
                  <div className="hero-bar-head">
                    <span><i style={{ background: '#6aa8ff' }} />Product B</span>
                    <span>20 units · $600</span>
                  </div>
                  <i className="track"><b style={{ '--w': '52%' } as CSSProperties} /></i>
                </div>
                <div className="hero-card-foot">
                  <ZapIcon /> Limiting resource: Machine · 100% utilized
                </div>
              </div>
            </div>
            <span className="float-chip fc-2"><CheckIcon /> HiGHS verified optimal</span>
          </div>
        </section>

        <section className="trust reveal">
          <span>BUILT FOR</span>
          <strong><SparkIcon /> Manufacturing</strong>
          <strong><SparkIcon /> Logistics</strong>
          <strong><SparkIcon /> Retail planning</strong>
          <strong><SparkIcon /> Studios &amp; agencies</strong>
          <strong><SparkIcon /> Founders</strong>
        </section>

        <section className="prompt-card reveal" id="solver">
          <div className="prompt-heading">
            <div>
              <h2>What would you like to optimize?</h2>
              <p>Use your own words — no fixed format needed. Production, transportation, shopping/purchase, or game-theory optimization only (not general math).</p>
            </div>
            <span className="badge">AI model selection + HiGHS solver</span>
          </div>

          <textarea
            value={problem}
            onChange={(event) => setProblem(event.target.value)}
            placeholder="Just describe your problem in your own words — e.g. 'I run a bakery making bread and cakes with limited oven time…' or 'I ship from 2 warehouses to 3 stores…' — I'll work out the details."
            maxLength={10000}
          />

          {trace && problem.trim().length >= 10 && !analysis && (
            <div className="trace-hint" role="status">
              <span className="trace-dot" aria-hidden />
              <span>
                {trace.missing.length === 0 ? (
                  <>Looks ready — detected <strong>{trace.detected}</strong>{trace.understood.length > 0 ? ` (${trace.understood.join('; ')})` : ''}. Hit “Solve my plan”.</>
                ) : (
                  <>Detected <strong>{trace.detected}</strong>{trace.understood.length > 0 ? ` — got ${trace.understood.join('; ')}` : ''}. Still need: <strong>{trace.missing.join(' + ')}</strong>.</>
                )}
              </span>
            </div>
          )}

          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button key={ex.label} type="button" className="example-chip" onClick={() => useExample(ex.prompt)}>
                <span className="ex-ico"><ExampleIcon kind={ex.icon} /></span>
                <span className="ex-text">
                  <strong>{ex.label}</strong>
                  <span>{ex.tag}</span>
                </span>
              </button>
            ))}
          </div>

          <div className="prompt-footer">
            <span>
              {charCount.toLocaleString()} / 10,000 chars · No signup · Any wording works ·{' '}
              {isAIConfigured() ? 'AI assist on' : 'Offline solver ready'}
            </span>
            <button className="btn btn-primary" id="solve-btn" onClick={handleSolve} disabled={loading} type="button">
              {loading ? (
                <>
                  <span className="spinner" /> Analyzing…
                </>
              ) : (
                <>
                  Solve my plan <span aria-hidden><ArrowIcon /></span>
                </>
              )}
            </button>
          </div>
        </section>

        {message && !clarify && (
          <p className="message error" role="alert">
            {message}
          </p>
        )}

        {clarify && (
          <section className="clarify-card" aria-live="polite">
            <p className="eyebrow">WHAT I UNDERSTOOD</p>
            <p className="clarify-text">{message}</p>
            <div className="clarify-grid">
              {clarify.understood.length > 0 && (
                <div>
                  <strong>Got it</strong>
                  <ul>
                    {clarify.understood.map((u) => (
                      <li key={u}>{u}</li>
                    ))}
                  </ul>
                </div>
              )}
              {clarify.missing.length > 0 && (
                <div>
                  <strong>Still need</strong>
                  <ul>
                    {clarify.missing.map((m) => (
                      <li key={m}>{m}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div className="clarify-actions">
              {(clarify.appendFix || clarify.fixExample) && (
                <button className="btn btn-primary btn-small" type="button" onClick={applyFix}>
                  Add what's missing &amp; solve
                </button>
              )}
              <button className="btn btn-ghost btn-small" type="button" onClick={useFullExample}>
                Show me a working example
              </button>
            </div>
            <small className="clarify-note">
              Tip: just keep typing in your own words — any phrasing with the same numbers works. No fixed format needed.
            </small>
          </section>
        )}

        {aiNote && solvedVia === 'browser' && (
          <p className="message" role="status">
            AI assist unavailable ({aiNote}) — answered with the offline solver instead.
          </p>
        )}

        {loading && (
          <section className="results skeleton" aria-label="Loading results">
            <div className="sk sk-title" />
            <div className="sk sk-line" />
            <div className="sk sk-line short" />
            <div className="result-grid">
              <div className="sk sk-box" />
              <div className="sk sk-box" />
            </div>
          </section>
        )}

        {analysis && !loading && (
          <section className="results" aria-live="polite" key={analysis.explanation}>
            <div className="result-panel formulation">
              <div className="formulation-head">
                <div>
                  <p className="eyebrow">MODEL UNDERSTANDING</p>
                  <h3>{analysis.problem_summary}</h3>
                </div>
                <div className="result-actions">
                  <span className="badge">
                    {solvedVia === 'api'
                      ? 'Solved via live API'
                      : solvedVia === 'ai'
                        ? 'Solved with AI'
                        : 'Solved in your browser'}
                  </span>
                  <button className="btn btn-ghost btn-small" onClick={copyPlan} type="button">
                    {copied ? 'Copied ✓' : 'Copy plan'}
                  </button>
                </div>
              </div>
              <div className="metric-row">
                <span>Technique</span>
                <strong>{analysis.technique}</strong>
              </div>
              <div className="metric-row">
                <span>Objective</span>
                <strong>{analysis.objective}</strong>
              </div>
              {analysis.variables.length > 0 && (
                <p>
                  <small>Decisions: {analysis.variables.join(', ')}</small>
                </p>
              )}
              {analysis.constraints.length > 0 && (
                <p>
                  <small>Constraints: {analysis.constraints.join('; ')}</small>
                </p>
              )}
              {analysis.assumptions.length > 0 && (
                <p>
                  <small>Assumptions: {analysis.assumptions.join('; ')}</small>
                </p>
              )}
            </div>

            {analysis.model_type === 'game_theory' ? (
              <>
                <div className="result-header">
                  <div>
                    <p className="eyebrow">GAME THEORY</p>
                    <h2>Recommended mixed strategy</h2>
                  </div>
                  <div className="profit">
                    <span>Expected payoff</span>
                    <strong><CountUp value={analysis.game_value ?? 0} decimals={4} trim /></strong>
                  </div>
                </div>
                <p className="explanation">{analysis.explanation}</p>
                <div className="result-panel">
                  <h3>Player 1 strategy</h3>
                  {analysis.strategies.map((strategy) => (
                    <div className="metric-row" key={strategy.action}>
                      <span>{strategy.action}</span>
                      <strong>{Math.round(strategy.probability * 100)}%</strong>
                    </div>
                  ))}
                </div>
              </>
            ) : analysis.model_type === 'transportation' ? (
              <>
                <div className="result-header">
                  <div>
                    <p className="eyebrow">TRANSPORTATION PLAN</p>
                    <h2>Shipping recommendation</h2>
                  </div>
                  <div className="profit">
                    <span>Total shipping cost</span>
                    <strong>
                      <CountUp value={analysis.total_cost ?? 0} decimals={2} prefix={analysis.currency ?? '$'} />
                    </strong>
                  </div>
                </div>
                <p className="explanation">{analysis.explanation}</p>
                <div className="result-panel">
                  <h3>Ship these quantities</h3>
                  {analysis.shipments.map((shipment) => (
                    <div className="metric-row" key={`${shipment.source}-${shipment.destination}`}>
                      <span>
                        {shipment.source} → {shipment.destination}
                      </span>
                      <strong>
                        {shipment.quantity} units · {analysis.currency ?? '$'}
                        {shipment.cost.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </strong>
                    </div>
                  ))}
                </div>
              </>
            ) : analysis.model_type === 'procurement' ? (
              <>
                <div className="result-header">
                  <div>
                    <p className="eyebrow">PURCHASE PLAN</p>
                    <h2>Where to buy</h2>
                  </div>
                  <div className="profit">
                    <span>Total cost{(analysis.discount_applied ?? 0) > 0 ? ' · discount applied' : ''}</span>
                    <strong>
                      <CountUp value={analysis.total_cost ?? 0} decimals={2} prefix={analysis.currency ?? '$'} />
                    </strong>
                  </div>
                </div>
                <p className="explanation">{analysis.explanation}</p>
                <div className="result-panel">
                  <h3>Buy this way</h3>
                  {analysis.purchases.map((purchase) => (
                    <div className="metric-row" key={`${purchase.shop}-${purchase.item}`}>
                      <span>
                        {purchase.shop} → {purchase.item}
                      </span>
                      <strong>
                        {purchase.quantity} units · {analysis.currency ?? '$'}
                        {purchase.cost.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </strong>
                    </div>
                  ))}
                  {(analysis.discount_applied ?? 0) > 0 && (
                    <div className="metric-row">
                      <span>Discount applied</span>
                      <strong className="hl">
                        −{analysis.currency ?? '$'}
                        {(analysis.discount_applied ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </strong>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                <div className="result-header">
                  <div>
                    <p className="eyebrow">OPTIMAL PLAN</p>
                    <h2>Production recommendation</h2>
                  </div>
                  <div className="profit">
                    <span>Total profit</span>
                    <strong>
                      <CountUp value={analysis.total_profit ?? 0} decimals={2} prefix={analysis.currency ?? '$'} />
                    </strong>
                  </div>
                </div>
                <p className="explanation">{analysis.explanation}</p>
                <div className="result-grid">
                  <div className="result-panel">
                    <h3>Make this much</h3>
                    {analysis.products.map((product) => (
                      <div className="metric-row" key={product.name}>
                        <span>{product.name}</span>
                        <strong>{product.quantity} units</strong>
                      </div>
                    ))}
                  </div>
                  <div className="result-panel">
                    <h3>Capacity used</h3>
                    {analysis.resources.map((resource) => (
                      <div className="resource" key={resource.name}>
                        <div className="metric-row">
                          <span>{resource.name}</span>
                          <strong className="hl">{Math.round(resource.utilization * 100)}%</strong>
                        </div>
                        <div className="bar">
                          <i style={{ '--w': `${Math.min(resource.utilization * 100, 100)}%` } as CSSProperties} />
                        </div>
                        <small>
                          {resource.used} of {resource.capacity} available
                        </small>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </section>
        )}

        <section className="features reveal" id="how">
          <article>
            <span className="step-ico"><PenIcon /></span>
            <span className="f-num">01</span>
            <h3>Describe</h3>
            <p>Write your planning problem naturally, just as you would explain it to a colleague. No math needed.</p>
          </article>
          <article>
            <span className="step-ico"><ZapIcon /></span>
            <span className="f-num">02</span>
            <h3>Optimize</h3>
            <p>AI classifies the model while the HiGHS solver finds the mathematically best plan — not a guess.</p>
          </article>
          <article>
            <span className="step-ico"><CheckIcon /></span>
            <span className="f-num">03</span>
            <h3>Decide</h3>
            <p>Get quantities, costs, utilization, and the reasoning behind them. Copy and share with your team.</p>
          </article>
        </section>

        <section className="use-cases reveal" id="use-cases">
          <p className="eyebrow">WHERE IT HELPS</p>
          <h2>One input box, four powerful solvers.</h2>
          <div className="use-grid">
            <div>
              <span className="use-ico"><FactoryIcon /></span>
              <h3>Production planning</h3>
              <p>Maximize profit under machine hours, labor, materials, and demand caps.</p>
            </div>
            <div>
              <span className="use-ico"><TruckIcon /></span>
              <h3>Transportation</h3>
              <p>Minimize shipping cost across warehouses, stores, supplies, and demands.</p>
            </div>
            <div>
              <span className="use-ico"><CartIcon /></span>
              <h3>Shopping &amp; purchase plans</h3>
              <p>Meet monthly needs at the lowest cost across shops, prices, and discounts.</p>
            </div>
            <div>
              <span className="use-ico"><GameIcon /></span>
              <h3>Game theory</h3>
              <p>Find optimal mixed strategies and guaranteed payoffs for competitive decisions.</p>
            </div>
          </div>
          <p className="note">
            All four model types understand your own wording instantly in your browser — no signup, no server, no API key.
            Describe profits with limited resources, shipping supplies/demands/costs, shopping needs with shop prices, or a game payoff. General math (equations, calculus) is out of scope.
            Connect the optional FastAPI backend with GPT-6 Astra for extra AI understanding of very long descriptions.
          </p>
        </section>
      </main>

      <footer>
        <div className="foot-inner">
          <div className="brand">
            <span className="brand-mark"><LogoIcon /></span> optisolve.ai
          </div>
          <p>Open-source operations research for everyone. Built with React, FastAPI, and HiGHS.</p>
          <div className="foot-links">
            <a href="https://github.com/bharath609/optisolve" target="_blank" rel="noreferrer">
              GitHub
            </a>
            <a href="https://bharath609.github.io/optisolve.ai/" target="_blank" rel="noreferrer">
              Live site
            </a>
            <a href="#solver">Solver</a>
          </div>
          <small>© 2026 OptiSolve · MIT License</small>
        </div>
      </footer>
    </div>
  )
}

export default App

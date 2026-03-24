interface Props {
  onReset: () => void
  onExampleQuery: (query: string) => void
}

const EXAMPLES = [
  "What's the EC number for chalcone isomerase?",
  'Tell me about lactate dehydrogenase',
  'Explain the catalytic mechanism of chymotrypsin',
  'What is the model architecture of RFDiffusion?',
  'Analyze lysozyme — structure and function',
]

const TOOLS = [
  { name: 'get_ec_number()', desc: 'EC classification' },
  { name: 'get_enzyme_structure()', desc: 'PDB data' },
  { name: 'get_catalytic_mechanism()', desc: 'Reaction details' },
  { name: 'search_research_papers()', desc: 'Hybrid RAG' },
  { name: 'search_uniprot()', desc: 'UniProt lookup' },
]

export function Sidebar({ onReset, onExampleQuery }: Props) {
  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar__brand">
        <HelixLogo />
        <div>
          <h1 className="sidebar__name">NeoBinder</h1>
          <p className="sidebar__tagline">Protein Design Agent</p>
        </div>
      </div>

      {/* Actions */}
      <div className="sidebar__actions">
        <button className="sidebar__btn" onClick={onReset}>
          <span className="sidebar__btn-icon">⊕</span>
          New Session
        </button>
      </div>

      {/* Examples */}
      <section className="sidebar__section">
        <h2 className="sidebar__section-label">EXAMPLES</h2>
        <ul className="sidebar__list">
          {EXAMPLES.map(q => (
            <li key={q}>
              <button className="sidebar__example" onClick={() => onExampleQuery(q)}>
                {q}
              </button>
            </li>
          ))}
        </ul>
      </section>

      {/* Tools */}
      <section className="sidebar__section">
        <h2 className="sidebar__section-label">TOOLS</h2>
        <ul className="sidebar__list">
          {TOOLS.map(t => (
            <li key={t.name} className="sidebar__tool">
              <code className="sidebar__tool-name">{t.name}</code>
              <span className="sidebar__tool-desc">{t.desc}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* Footer */}
      <div className="sidebar__footer">
        <span className="sidebar__pulse" />
        <span className="sidebar__footer-text">Agent online</span>
      </div>
    </aside>
  )
}

function HelixLogo() {
  return (
    <svg
      className="sidebar__logo"
      width="38"
      height="38"
      viewBox="0 0 38 38"
      fill="none"
      aria-hidden="true"
    >
      {/* Outer ring */}
      <circle cx="19" cy="19" r="18" stroke="var(--accent)" strokeWidth="1" opacity="0.25" />
      {/* Strand A */}
      <path
        d="M19 5 C10 10, 28 16, 19 21 C10 26, 28 32, 19 37"
        stroke="var(--accent)"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
      />
      {/* Strand B */}
      <path
        d="M19 5 C28 10, 10 16, 19 21 C28 26, 10 32, 19 37"
        stroke="var(--accent-2)"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        opacity="0.65"
      />
      {/* Rungs */}
      <line x1="13" y1="13" x2="25" y2="13" stroke="var(--accent)" strokeWidth="0.75" opacity="0.4" />
      <line x1="11" y1="19" x2="27" y2="19" stroke="var(--accent)" strokeWidth="0.75" opacity="0.4" />
      <line x1="13" y1="25" x2="25" y2="25" stroke="var(--accent)" strokeWidth="0.75" opacity="0.4" />
    </svg>
  )
}

import { useEffect, useRef, useState, useCallback } from 'react'

/* ── 3Dmol.js global types ───────────────────────────────── */
declare global {
  interface Window {
    $3Dmol: {
      createViewer: (el: HTMLElement, config?: Record<string, unknown>) => Viewer3D
      download: (
        query: string,
        viewer: Viewer3D,
        options: Record<string, unknown>,
        callback?: () => void,
      ) => void
    }
  }
}

interface Viewer3D {
  setStyle: (sel: Record<string, unknown>, style: Record<string, unknown>) => void
  addStyle: (sel: Record<string, unknown>, style: Record<string, unknown>) => void
  selectedAtoms: (sel: Record<string, unknown>) => AtomSpec[]
  zoomTo: (sel?: Record<string, unknown>) => void
  render: () => void
  clear: () => void
  removeAllModels: () => void
  resize: () => void
}

interface AtomSpec {
  chain: string
  resi: number
  resn: string
  atom: string
}

/* ── Residue data ────────────────────────────────────────── */
type ResType = 'hydrophobic' | 'polar' | 'charged-pos' | 'charged-neg' | 'special'

interface ResidueInfo {
  chain: string
  resi: number
  resn: string
  oneLetterCode: string
  type: ResType
}

const THREE_TO_ONE: Record<string, string> = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C',
  GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P',
  SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  // common non-standard
  MSE: 'M', HYP: 'P', CSO: 'C', SEC: 'U', PYL: 'O',
}

const RES_TYPE: Record<string, ResType> = {
  A: 'hydrophobic', V: 'hydrophobic', I: 'hydrophobic', L: 'hydrophobic',
  M: 'hydrophobic', F: 'hydrophobic', W: 'hydrophobic', P: 'hydrophobic',
  S: 'polar', T: 'polar', N: 'polar', Q: 'polar', C: 'polar',
  Y: 'polar', U: 'polar',
  R: 'charged-pos', K: 'charged-pos', H: 'charged-pos',
  D: 'charged-neg', E: 'charged-neg',
  G: 'special',
}

const RAINBOW_STYLE = {
  cartoon: { colorscheme: { gradient: 'rainbow', min: 1, max: 500 } },
}

/* ── Component ───────────────────────────────────────────── */
interface Props {
  pdbId: string
  onClose: () => void
}

const MIN_W = 280
const MAX_W = 900
const DEFAULT_W = 360

export function ProteinViewer({ pdbId, onClose }: Props) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<Viewer3D | null>(null)
  const isDragging = useRef(false)
  const dragStartX = useRef(0)
  const dragStartW = useRef(DEFAULT_W)

  const [panelWidth, setPanelWidth] = useState(DEFAULT_W)
  const [isResizing, setIsResizing] = useState(false)
  const [residues, setResidues] = useState<ResidueInfo[]>([])
  const [chains, setChains] = useState<string[]>([])
  const [activeChain, setActiveChain] = useState<string>('')
  const [selected, setSelected] = useState<{ chain: string; resi: number } | null>(null)
  const [hovered, setHovered] = useState<ResidueInfo | null>(null)
  const [loading, setLoading] = useState(true)

  /* ── Apply highlight / reset ─────────────────────────── */
  const applyStyle = useCallback(
    (sel: { chain: string; resi: number } | null) => {
      const v = viewerRef.current
      if (!v) return
      v.setStyle({}, RAINBOW_STYLE)
      if (sel) {
        v.addStyle(
          { resi: sel.resi, chain: sel.chain },
          { sphere: { color: '#FFD700', radius: 1.0, opacity: 0.85 } },
        )
        v.addStyle(
          { resi: sel.resi, chain: sel.chain },
          { stick: { colorscheme: 'Jmol', radius: 0.15 } },
        )
      }
      v.render()
    },
    [],
  )

  /* ── Click residue chip ──────────────────────────────── */
  const handleResidueClick = useCallback(
    (res: ResidueInfo) => {
      const isSame = selected?.chain === res.chain && selected?.resi === res.resi
      const next = isSame ? null : { chain: res.chain, resi: res.resi }
      setSelected(next)
      applyStyle(next)
    },
    [selected, applyStyle],
  )

  /* ── Extract sequence from loaded model ──────────────── */
  const extractSequence = useCallback(() => {
    const v = viewerRef.current
    if (!v) return
    const atoms = v.selectedAtoms({ atom: 'CA' }) as AtomSpec[]
    const seen = new Set<string>()
    const list: ResidueInfo[] = []
    for (const a of atoms) {
      const key = `${a.chain}:${a.resi}`
      if (seen.has(key)) continue
      seen.add(key)
      const one = THREE_TO_ONE[a.resn] ?? '?'
      list.push({
        chain: a.chain,
        resi: a.resi,
        resn: a.resn,
        oneLetterCode: one,
        type: RES_TYPE[one] ?? 'special',
      })
    }
    list.sort((a, b) => a.chain.localeCompare(b.chain) || a.resi - b.resi)
    setResidues(list)
    const uniqueChains = [...new Set(list.map(r => r.chain))]
    setChains(uniqueChains)
    setActiveChain(uniqueChains[0] ?? '')
  }, [])

  /* ── Init / reload 3Dmol on pdbId change ────────────── */
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setResidues([])
    setChains([])
    setSelected(null)

    const init = () => {
      if (cancelled || !canvasRef.current || !window.$3Dmol) return
      try {
        if (!viewerRef.current) {
          viewerRef.current = window.$3Dmol.createViewer(canvasRef.current, {
            backgroundColor: '#07090F',
            antialias: true,
          })
        } else {
          viewerRef.current.removeAllModels()
        }
        window.$3Dmol.download(`pdb:${pdbId}`, viewerRef.current, {}, () => {
          if (cancelled || !viewerRef.current) return
          viewerRef.current.setStyle({}, RAINBOW_STYLE)
          viewerRef.current.zoomTo()
          // Let the layout settle before first render so canvas has real dimensions
          requestAnimationFrame(() => {
            if (cancelled || !viewerRef.current) return
            viewerRef.current.resize()
            viewerRef.current.render()
            extractSequence()
            setLoading(false)
          })
        })
      } catch (e) {
        console.error('3Dmol error:', e)
        setLoading(false)
      }
    }

    if (window.$3Dmol) {
      init()
    } else {
      const t = setInterval(() => {
        if (window.$3Dmol) { clearInterval(t); init() }
      }, 200)
      return () => { cancelled = true; clearInterval(t) }
    }
    return () => { cancelled = true }
  }, [pdbId, extractSequence])

  /* ── ResizeObserver: keep 3Dmol in sync with panel size ─ */
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      viewerRef.current?.resize()
      viewerRef.current?.render()
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  /* ── Drag-to-resize ──────────────────────────────────── */
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDragging.current) return
      const dx = dragStartX.current - e.clientX
      const w = Math.min(Math.max(dragStartW.current + dx, MIN_W), MAX_W)
      setPanelWidth(w)
    }
    const onUp = () => {
      if (!isDragging.current) return
      isDragging.current = false
      setIsResizing(false)
      viewerRef.current?.resize()
      viewerRef.current?.render()
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
    return () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
  }, [])

  const handleResizeMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true
    dragStartX.current = e.clientX
    dragStartW.current = panelWidth
    setIsResizing(true)
  }

  /* ── Filtered residues for active chain ──────────────── */
  const chainResidues = residues.filter(r => r.chain === activeChain)

  return (
    <aside
      className={`viewer-panel${isResizing ? ' viewer-panel--resizing' : ''}`}
      style={{ width: panelWidth }}
    >
      {/* Drag handle */}
      <div
        className={`viewer-panel__resize-handle${isResizing ? ' viewer-panel__resize-handle--active' : ''}`}
        onMouseDown={handleResizeMouseDown}
        title="Drag to resize"
      />

      {/* Header */}
      <div className="viewer-panel__header">
        <div className="viewer-panel__title">
          <span className="viewer-panel__glyph">⬡</span>
          <span>{pdbId}</span>
          <span className="viewer-panel__subtitle">Structure</span>
        </div>
        <button className="viewer-panel__close" onClick={onClose} aria-label="Close viewer">
          ✕
        </button>
      </div>

      {/* 3Dmol canvas */}
      <div className="viewer-panel__canvas" ref={canvasRef}>
        {loading && (
          <div className="viewer-panel__loading">
            <span className="spinner" />
            <span>Loading {pdbId}…</span>
          </div>
        )}
      </div>

      {/* Sequence viewer */}
      {residues.length > 0 && (
        <div className="seq-viewer">
          {/* Chain tabs */}
          {chains.length > 1 && (
            <div className="seq-viewer__chains">
              {chains.map(c => (
                <button
                  key={c}
                  className={`seq-viewer__chain-btn${activeChain === c ? ' seq-viewer__chain-btn--active' : ''}`}
                  onClick={() => setActiveChain(c)}
                >
                  Chain {c}
                </button>
              ))}
            </div>
          )}

          {/* Residue strip */}
          <div className="seq-viewer__strip">
            {chainResidues.map(res => {
              const isSel = selected?.chain === res.chain && selected?.resi === res.resi
              return (
                <button
                  key={`${res.chain}-${res.resi}`}
                  className={`seq-viewer__residue seq-viewer__residue--${res.type}${isSel ? ' seq-viewer__residue--selected' : ''}`}
                  onClick={() => handleResidueClick(res)}
                  onMouseEnter={() => setHovered(res)}
                  onMouseLeave={() => setHovered(null)}
                  title={`${res.resn} ${res.resi} (Chain ${res.chain})`}
                >
                  {res.oneLetterCode}
                </button>
              )
            })}
          </div>

          {/* Info bar */}
          <div className="seq-viewer__info">
            {hovered ? (
              <>
                <span className="seq-viewer__info-name">{hovered.resn}</span>
                <span className="seq-viewer__info-sep">·</span>
                <span>Residue {hovered.resi}</span>
                <span className="seq-viewer__info-sep">·</span>
                <span>Chain {hovered.chain}</span>
                {selected?.chain === hovered.chain && selected?.resi === hovered.resi && (
                  <span className="seq-viewer__info-sel">● selected</span>
                )}
              </>
            ) : selected ? (
              <>
                <span className="seq-viewer__info-sel">● </span>
                <span>
                  Residue {selected.resi} · Chain {selected.chain} — click again to deselect
                </span>
              </>
            ) : (
              <span className="seq-viewer__info-hint">
                {chainResidues.length} residues · click to highlight
              </span>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="viewer-panel__footer">
        <span className="viewer-panel__id">PDB: {pdbId}</span>
        <a
          href={`https://www.rcsb.org/structure/${pdbId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="viewer-panel__link"
        >
          View on RCSB →
        </a>
      </div>
    </aside>
  )
}

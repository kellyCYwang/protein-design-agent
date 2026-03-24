import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'
import type { Message } from '../types'

interface Props {
  message: Message
  onPdbOpen: (id: string) => void
}

const BADGE_LABELS: Record<string, string> = {
  simple: 'EC Lookup',
  detailed: 'Analysis',
  research: 'Research',
}

// All patterns that the LLM might use to cite a PDB ID
const PDB_SCAN_RE =
  /\bPDB(?:\s+(?:ID|code|entry|accession))?\b[^A-Z0-9]{0,15}([A-Z0-9]{4})\b/gi

function scanPdbIds(text: string): string[] {
  const ids = new Set<string>()
  let m: RegExpExecArray | null
  // Reset lastIndex since the flag 'g' keeps state
  PDB_SCAN_RE.lastIndex = 0
  while ((m = PDB_SCAN_RE.exec(text)) !== null) {
    ids.add(m[1].toUpperCase())
  }
  return [...ids]
}

// Custom inline code renderer — makes bare 4-char alphanumeric codes clickable
function makePdbCodeComponent(
  onPdbOpen: (id: string) => void,
): Components['code'] {
  return function PdbCode({ children, ...rest }) {
    const text = String(children).trim()
    if (/^[A-Z0-9]{4}$/i.test(text)) {
      return (
        <code
          {...rest}
          className="message__pdb-inline"
          onClick={() => onPdbOpen(text.toUpperCase())}
          title={`Click to view 3D structure for ${text.toUpperCase()}`}
        >
          {text.toUpperCase()}
        </code>
      )
    }
    return <code {...rest}>{children}</code>
  }
}

export function MessageBubble({ message, onPdbOpen }: Props) {
  const isUser = message.role === 'user'
  const isEmpty = !message.content

  const timeLabel = message.timestamp.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  })

  // Scan for PDB IDs to show view-structure buttons
  const pdbIds = !isUser && message.content ? scanPdbIds(message.content) : []

  const components: Components = {
    code: makePdbCodeComponent(onPdbOpen),
  }

  return (
    <div className={`message message--${isUser ? 'user' : 'assistant'}`}>
      <div className="message__meta">
        <span className="message__role">{isUser ? 'YOU' : 'NEOBINDER'}</span>

        {!isUser && message.queryType && (
          <span className={`message__badge message__badge--${message.queryType}`}>
            {BADGE_LABELS[message.queryType] ?? message.queryType}
          </span>
        )}

        {!isUser && message.toolsUsed && message.toolsUsed.length > 0 && (
          <span className="message__tools">
            {message.toolsUsed.map(t => (
              <code key={t} className="message__tool-chip">
                {t}
              </code>
            ))}
          </span>
        )}

        <span className="message__time">{timeLabel}</span>
      </div>

      <div className="message__bubble">
        {isEmpty ? (
          <div className="message__thinking">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        ) : isUser ? (
          <p className="message__text">{message.content}</p>
        ) : (
          <div className="message__markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {/* View-structure buttons for every detected PDB ID */}
      {pdbIds.length > 0 && (
        <div className="message__pdb-actions">
          {pdbIds.map(id => (
            <button
              key={id}
              className="message__pdb-btn"
              onClick={() => onPdbOpen(id)}
            >
              <span className="message__pdb-btn-glyph">⬡</span>
              View {id}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

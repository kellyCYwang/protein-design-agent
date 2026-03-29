import { useState, useEffect, useMemo } from 'react'
import { Sidebar } from './components/Sidebar'
import { ChatArea } from './components/ChatArea'
import { PipelineStatus } from './components/PipelineStatus'
import { ProteinViewer } from './components/ProteinViewer'
import { useChat } from './hooks/useChat'

// Catches all common LLM formats:
//   "PDB ID: 1ABC"  "PDB: 1ABC"  "**PDB ID**: 1ABC"
//   "PDB code 1ABC"  "PDB entry: 1abc"  "PDB ID is 1ABC"
const PDB_RE = /\bPDB(?:\s+(?:ID|code|entry|accession))?\b[^A-Z0-9]{0,15}([A-Z0-9]{4})\b/i

function detectPdbId(text: string): string | null {
  const m = PDB_RE.exec(text)
  return m ? m[1].toUpperCase() : null
}

export default function App() {
  const { messages, status, isStreaming, send, cancel, reset } = useChat()
  const [activePdbId, setActivePdbId] = useState<string | null>(null)
  const [showViewer, setShowViewer] = useState(false)

  // Auto-detect from the latest assistant message
  const detectedPdbId = useMemo(() => {
    const last = [...messages].reverse().find(m => m.role === 'assistant')
    if (!last?.content) return null
    return detectPdbId(last.content)
  }, [messages])

  useEffect(() => {
    if (detectedPdbId) {
      setActivePdbId(detectedPdbId)
      setShowViewer(true)
    }
  }, [detectedPdbId])

  const handlePdbOpen = (id: string) => {
    setActivePdbId(id)
    setShowViewer(true)
  }

  const handleClose = () => setShowViewer(false)

  return (
    <div className={`app${showViewer && activePdbId ? ' app--viewer-open' : ''}`}>
      <Sidebar onReset={reset} onExampleQuery={send} />

      <div className="app__main">
        <PipelineStatus status={status} />
        <ChatArea
          messages={messages}
          isStreaming={isStreaming}
          onSend={send}
          onCancel={cancel}
          onPdbOpen={handlePdbOpen}
        />
      </div>

      {showViewer && activePdbId && (
        <ProteinViewer pdbId={activePdbId} onClose={handleClose} />
      )}
    </div>
  )
}

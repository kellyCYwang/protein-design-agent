import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import type { Message } from '../types'
import { MessageBubble } from './MessageBubble'

interface Props {
  messages: Message[]
  isStreaming: boolean
  onSend: (content: string) => void
  onPdbOpen: (id: string) => void
}

export function ChatArea({ messages, isStreaming, onSend, onPdbOpen }: Props) {
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-scroll on new messages
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`
  }, [input])

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    setInput('')
    onSend(trimmed)
  }

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat">
      <div className="chat__messages" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="chat__empty">
            <svg className="chat__empty-glyph" viewBox="0 0 60 60" fill="none">
              <circle cx="30" cy="30" r="28" stroke="currentColor" strokeWidth="1" opacity="0.2" />
              <path
                d="M30 8 C18 14, 42 22, 30 30 C18 38, 42 46, 30 52"
                stroke="currentColor"
                strokeWidth="2"
                fill="none"
                strokeLinecap="round"
              />
              <path
                d="M30 8 C42 14, 18 22, 30 30 C42 38, 18 46, 30 52"
                stroke="currentColor"
                strokeWidth="2"
                fill="none"
                strokeLinecap="round"
                opacity="0.5"
              />
            </svg>
            <p className="chat__empty-title">Begin your analysis</p>
            <p className="chat__empty-sub">
              Ask about enzymes, protein structures, reaction mechanisms, or research papers.
            </p>
          </div>
        ) : (
          messages.map(m => <MessageBubble key={m.id} message={m} onPdbOpen={onPdbOpen} />)
        )}
      </div>

      <div className="chat__input-row">
        <div className="chat__input-wrap">
          <textarea
            ref={textareaRef}
            className="chat__input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about an enzyme, reaction mechanism, or protein structure…"
            rows={1}
            disabled={isStreaming}
          />
          <div className="chat__input-hint">↵ send &nbsp;·&nbsp; ⇧↵ newline</div>
        </div>
        <button
          className={`chat__send${isStreaming ? ' chat__send--loading' : ''}`}
          onClick={handleSend}
          disabled={isStreaming || !input.trim()}
          aria-label="Send message"
        >
          {isStreaming ? (
            <span className="spinner" />
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path
                d="M2 8h12M10 4l4 4-4 4"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}

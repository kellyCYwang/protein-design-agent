import { useState, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'
import type { Message, AgentStatus, QueryType } from '../types'

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [status, setStatus] = useState<AgentStatus>({ stage: 'idle' })
  const [isStreaming, setIsStreaming] = useState(false)
  const threadIdRef = useRef<string>(uuidv4())
  const abortRef = useRef<AbortController | null>(null)

  const cancel = useCallback(async () => {
    // 1. Abort the fetch (closes the SSE stream client-side)
    abortRef.current?.abort()
    abortRef.current = null

    // 2. Tell the backend to stop the agent thread
    try {
      await fetch('/api/chat/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadIdRef.current }),
      })
    } catch {
      // Best-effort — backend may already be done
    }

    // 3. Update UI state
    setIsStreaming(false)
    setStatus({ stage: 'idle' })

    // 4. Mark the last assistant message as cancelled
    setMessages(prev => {
      const last = prev[prev.length - 1]
      if (last?.role === 'assistant' && !last.content) {
        // Empty placeholder — replace with cancelled notice
        return [
          ...prev.slice(0, -1),
          { ...last, content: '*Query cancelled.*' },
        ]
      }
      if (last?.role === 'assistant') {
        // Partial response — append cancellation note
        return [
          ...prev.slice(0, -1),
          { ...last, content: last.content + '\n\n*— cancelled*' },
        ]
      }
      return prev
    })
  }, [])

  const send = useCallback(
    async (content: string) => {
      if (isStreaming) return

      const userMessage: Message = {
        id: uuidv4(),
        role: 'user',
        content,
        timestamp: new Date(),
      }

      setMessages(prev => [...prev, userMessage])
      setIsStreaming(true)
      setStatus({ stage: 'routing' })

      const assistantId = uuidv4()
      const toolsUsed: string[] = []
      let queryType: QueryType | undefined

      // Placeholder so the thinking animation appears immediately
      setMessages(prev => [
        ...prev,
        { id: assistantId, role: 'assistant', content: '', timestamp: new Date() },
      ])

      // Create abort controller for this request
      const abortController = new AbortController()
      abortRef.current = abortController

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: content, thread_id: threadIdRef.current }),
          signal: abortController.signal,
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const reader = response.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let currentEvent = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            const trimmed = line.trim()
            if (trimmed.startsWith('event: ')) {
              currentEvent = trimmed.slice(7).trim()
            } else if (trimmed.startsWith('data: ') && currentEvent) {
              try {
                const data = JSON.parse(trimmed.slice(6))
                switch (currentEvent) {
                  case 'route':
                    queryType = data.query_type as QueryType
                    setStatus({ stage: 'routing', queryType })
                    break
                  case 'status':
                    setStatus({ stage: 'rag' })
                    break
                  case 'tool':
                    toolsUsed.push(data.tool_name as string)
                    setStatus({ stage: 'tool', toolName: data.tool_name as string })
                    break
                  case 'response':
                    setStatus({ stage: 'responding', queryType })
                    setMessages(prev =>
                      prev.map(m =>
                        m.id === assistantId
                          ? {
                              ...m,
                              content: data.content as string,
                              queryType,
                              toolsUsed: [...toolsUsed],
                            }
                          : m,
                      ),
                    )
                    break
                  case 'cancelled':
                    setStatus({ stage: 'idle' })
                    break
                  case 'done':
                    setStatus({ stage: 'done' })
                    break
                  case 'error':
                    setStatus({ stage: 'error', errorMessage: data.message as string })
                    setMessages(prev =>
                      prev.map(m =>
                        m.id === assistantId
                          ? { ...m, content: `**Error:** ${data.message as string}` }
                          : m,
                      ),
                    )
                    break
                }
                currentEvent = ''
              } catch {
                // Ignore malformed SSE lines
              }
            }
          }
        }
      } catch (err) {
        // Don't show errors for intentional aborts
        if (err instanceof DOMException && err.name === 'AbortError') {
          return
        }
        const msg = err instanceof Error ? err.message : 'Unknown error'
        setStatus({ stage: 'error', errorMessage: msg })
        setMessages(prev =>
          prev.map(m => (m.id === assistantId ? { ...m, content: `**Error:** ${msg}` } : m)),
        )
      } finally {
        abortRef.current = null
        setIsStreaming(false)
        setTimeout(() => setStatus({ stage: 'idle' }), 3000)
      }
    },
    [isStreaming],
  )

  const reset = useCallback(() => {
    // Cancel any in-flight request before resetting
    abortRef.current?.abort()
    abortRef.current = null
    setMessages([])
    setStatus({ stage: 'idle' })
    setIsStreaming(false)
    threadIdRef.current = uuidv4()
  }, [])

  return {
    messages,
    status,
    isStreaming,
    send,
    cancel,
    reset,
    threadId: threadIdRef.current,
  }
}

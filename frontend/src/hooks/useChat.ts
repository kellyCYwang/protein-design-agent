import { useState, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'
import type { Message, AgentStatus, QueryType } from '../types'

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [status, setStatus] = useState<AgentStatus>({ stage: 'idle' })
  const [isStreaming, setIsStreaming] = useState(false)
  const threadIdRef = useRef<string>(uuidv4())

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

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: content, thread_id: threadIdRef.current }),
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
        const msg = err instanceof Error ? err.message : 'Unknown error'
        setStatus({ stage: 'error', errorMessage: msg })
        setMessages(prev =>
          prev.map(m => (m.id === assistantId ? { ...m, content: `**Error:** ${msg}` } : m)),
        )
      } finally {
        setIsStreaming(false)
        setTimeout(() => setStatus({ stage: 'idle' }), 3000)
      }
    },
    [isStreaming],
  )

  const reset = useCallback(() => {
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
    reset,
    threadId: threadIdRef.current,
  }
}

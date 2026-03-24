export type QueryType = 'simple' | 'detailed' | 'research'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  queryType?: QueryType
  toolsUsed?: string[]
}

export type AgentStage = 'idle' | 'routing' | 'rag' | 'tool' | 'responding' | 'done' | 'error'

export interface AgentStatus {
  stage: AgentStage
  queryType?: QueryType
  toolName?: string
  errorMessage?: string
}

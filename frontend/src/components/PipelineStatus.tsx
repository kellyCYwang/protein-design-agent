import type { AgentStatus } from '../types'

interface Props {
  status: AgentStatus
}

const PIPELINE_STAGES = [
  { id: 'routing', label: 'Route' },
  { id: 'rag', label: 'Research' },
  { id: 'tool', label: 'Tool' },
  { id: 'responding', label: 'Respond' },
] as const

const STAGE_ORDER = ['idle', 'routing', 'rag', 'tool', 'responding', 'done', 'error']

export function PipelineStatus({ status }: Props) {
  if (status.stage === 'idle') return null

  const currentIdx = STAGE_ORDER.indexOf(status.stage)
  const isError = status.stage === 'error'

  return (
    <div className={`pipeline${isError ? ' pipeline--error' : ''}`}>
      <div className="pipeline__stages">
        {PIPELINE_STAGES.map((stage, i) => {
          const stageIdx = STAGE_ORDER.indexOf(stage.id)
          const isDone = currentIdx > stageIdx && !isError
          const isActive = status.stage === stage.id

          const label =
            isActive && stage.id === 'tool' && status.toolName
              ? status.toolName
              : stage.label

          return (
            <div key={stage.id} className="pipeline__item">
              <div
                className={`pipeline__node${isDone ? ' pipeline__node--done' : ''}${isActive ? ' pipeline__node--active' : ''}`}
              >
                {isDone ? '✓' : i + 1}
              </div>
              <span className={`pipeline__label${isActive ? ' pipeline__label--active' : ''}`}>
                {label}
              </span>
              {i < PIPELINE_STAGES.length - 1 && (
                <div className={`pipeline__connector${isDone ? ' pipeline__connector--done' : ''}`} />
              )}
            </div>
          )
        })}
      </div>
      {isError && <span className="pipeline__error-msg">{status.errorMessage}</span>}
    </div>
  )
}

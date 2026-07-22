import { useEffect, useRef, useState, useEffectEvent } from 'react'

const BACKEND_URL = (
  import.meta.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

function formatTime(seconds) {
  const safeSeconds = Math.max(0, seconds)
  const minutes = Math.floor(safeSeconds / 60)
  const remainder = safeSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

export default function App() {
  const audioRef = useRef(null)
  const objectUrlRef = useRef(null)
  const [selectedPresetIndex, setSelectedPresetIndex] = useState(0)
  const [presets, setPresets] = useState([]) // Array of { name, label, expression }
  const [preparedPresets, setPreparedPresets] = useState([])
  const [playbackOffset, setPlaybackOffset] = useState(0)
  const [durationSeconds, setDurationSeconds] = useState(0)
  const [status, setStatus] = useState('Preparing presets')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentExpression, setCurrentExpression] = useState('')
  const [expressionInput, setExpressionInput] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [seekPosition, setSeekPosition] = useState(0)

  const selectedPreset = presets[selectedPresetIndex] || { name: 'none', label: 'no preset', expression: '' }

  const releaseObjectUrl = useEffectEvent(() => {
    if (objectUrlRef.current !== null) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
  })

  const absolutePlaybackTime = useEffectEvent(() => {
    const audioElement = audioRef.current
    return audioElement === null ? playbackOffset : playbackOffset + audioElement.currentTime
  })

  const fetchSlice = useEffectEvent(async (preset, fromTime, autoplay) => {
    setStatus(`Loading ${preset.label} from ${formatTime(fromTime)}`)
    const response = await fetch(`${BACKEND_URL}/api/audio-slice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset_name: preset.name, from_time: fromTime }),
    })
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || `Backend request failed with ${response.status}`)
    }

    const audioBlob = await response.blob()
    const nextObjectUrl = URL.createObjectURL(audioBlob)
    const audioElement = audioRef.current
    if (audioElement === null) {
      URL.revokeObjectURL(nextObjectUrl)
      return
    }

    releaseObjectUrl()
    objectUrlRef.current = nextObjectUrl
    audioElement.src = nextObjectUrl
    audioElement.currentTime = 0
    setPlaybackOffset(fromTime)
    setSeekPosition(fromTime)
    
    if (autoplay) {
      await audioElement.play()
      setIsPlaying(true)
      setStatus(`Playing ${preset.label} from ${formatTime(fromTime)}`)
    } else {
      setIsPlaying(false)
      setStatus(`Ready on ${preset.label} at ${formatTime(fromTime)}`)
    }
  })

  const startPlayback = useEffectEvent(async () => {
    const audioElement = audioRef.current
    if (audioElement !== null) audioElement.pause()
    await fetchSlice(selectedPreset, 0, true)
  })

  const stopPlayback = useEffectEvent(() => {
    const audioElement = audioRef.current
    if (audioElement === null) return
    const nextTime = absolutePlaybackTime()
    audioElement.pause()
    setPlaybackOffset(nextTime)
    setSeekPosition(nextTime)
    setIsPlaying(false)
    setStatus(`Stopped at ${formatTime(nextTime)}`)
  })

  const selectPreset = useEffectEvent(async (presetIndex) => {
    if (!presets[presetIndex]) return
    const preset = presets[presetIndex]
    const nextTime = absolutePlaybackTime()
    const shouldAutoplay = isPlaying
    const audioElement = audioRef.current
    if (audioElement !== null) audioElement.pause()
    setSelectedPresetIndex(presetIndex)
    await fetchSlice(preset, nextTime, shouldAutoplay)
  })

  const handleSeek = useEffectEvent((event) => {
    const newSeekPosition = parseFloat(event.target.value)
    setSeekPosition(newSeekPosition)
    const audioElement = audioRef.current
    if (audioElement !== null) {
      audioElement.currentTime = newSeekPosition - playbackOffset
    }
  })

  const handleSeekCommit = useEffectEvent(() => {
    const audioElement = audioRef.current
    if (audioElement === null) return
    const shouldAutoplay = isPlaying
    audioElement.pause()
    audioElement.currentTime = seekPosition - playbackOffset
    if (shouldAutoplay) {
      audioElement.play()
      setIsPlaying(true)
    }
    setStatus(`Seeked to ${formatTime(seekPosition)}`)
  })

  const submitExpression = useEffectEvent(async () => {
    if (!expressionInput.trim() || isSubmitting) return
    
    setIsSubmitting(true)
    const audioElement = audioRef.current
    const currentTime = absolutePlaybackTime()
    
    if (audioElement !== null) audioElement.pause()

    try {
      const response = await fetch(`${BACKEND_URL}/api/apply-expression`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression: expressionInput.trim(), from_time: currentTime }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `Backend request failed with ${response.status}`)
      }

      setCurrentExpression(expressionInput.trim())
      setExpressionInput('')
      
      // Restart audio from current time with the new expression applied
      await fetchSlice(selectedPreset, currentTime, isPlaying)
      setStatus(`Expression applied at ${formatTime(currentTime)}`)
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error))
    } finally {
      setIsSubmitting(false)
    }
  })

  const copyExpression = useEffectEvent(async () => {
    if (!currentExpression) return
    try {
      await navigator.clipboard.writeText(currentExpression)
      setStatus('Expression copied to clipboard')
    } catch {
      setStatus('Failed to copy expression')
    }
  })

  const handleKeydown = useEffectEvent(async (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.repeat) return

    try {
      if (event.key === 'p' || event.key === 'P') {
        event.preventDefault()
        await startPlayback()
      } else if (event.key === 's' || event.key === 'S') {
        event.preventDefault()
        stopPlayback()
      } else if (event.key >= '0' && event.key < String(presets.length)) {
        event.preventDefault()
        await selectPreset(Number(event.key))
      }
    } catch (error) {
      setIsPlaying(false)
      setStatus(error instanceof Error ? error.message : String(error))
    }
  })

  useEffect(() => {
    const audioElement = audioRef.current
    if (audioElement === null) return undefined

    const handleTimeUpdate = () => {
      const audioElement = audioRef.current
      if (audioElement !== null) {
        setSeekPosition(absolutePlaybackTime())
      }
    }

    const handleEnded = () => {
      const nextTime = absolutePlaybackTime()
      setPlaybackOffset(nextTime)
      setSeekPosition(nextTime)
      setIsPlaying(false)
      setStatus(`Finished at ${formatTime(nextTime)}`)
    }

    audioElement.addEventListener('ended', handleEnded)
    audioElement.addEventListener('timeupdate', handleTimeUpdate)
    return () => {
      audioElement.removeEventListener('ended', handleEnded)
      audioElement.removeEventListener('timeupdate', handleTimeUpdate)
    }
  }, [])

  useEffect(() => {
    let active = true

    async function loadPresetData() {
      try {
        // Fetch preset expressions from backend
        const exprResponse = await fetch(`${BACKEND_URL}/api/presets/expressions`)
        if (!exprResponse.ok) {
          throw new Error(`Failed to fetch expressions: ${exprResponse.status}`)
        }
        const expressionMap = await exprResponse.json()
        
        // Build presets array with name, label, and expression
        const presetList = Object.entries(expressionMap).map(([name, expression]) => ({
          name,
          label: name.replace(/([A-Z])/g, ' $1').trim(), // Pretty label
          expression,
        }))
        
        setPresets(presetList)

        // Prepare the presets (send preset_names to backend for caching/rendering)
        const prepResponse = await fetch(`${BACKEND_URL}/api/presets/prepare`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ preset_names: presetList.map(p => p.name) }),
        })
        if (!prepResponse.ok) {
          const errorText = await prepResponse.text()
          throw new Error(errorText || `Backend request failed with ${prepResponse.status}`)
        }
        const payload = await prepResponse.json()
        
        setPreparedPresets(payload.preset_names)
        setDurationSeconds(payload.duration_seconds)
        
        if (active) {
          setStatus(`Ready on ${presetList[0]?.label || 'none'}`)
        }
      } catch (error) {
        if (!active) return
        setStatus(error instanceof Error ? error.message : String(error))
      }
    }

    loadPresetData()

    const listener = (event) => void handleKeydown(event)
    window.addEventListener('keydown', listener)
    return () => {
      active = false
      window.removeEventListener('keydown', listener)
      releaseObjectUrl()
    }
  }, [])

  return (
    <main className="app-shell">
      <header className="hero">
        <p className="eyebrow">Preset Preview</p>
        <h1>Radio Drama Preset Switcher</h1>
        <p className="summary">
          Keyboard only. Press p to restart playback, s to stop, and 0 through 6 to
          swap presets at the current production time.
        </p>
      </header>

      <section className="status-panel" aria-live="polite" aria-atomic="true">
        <p className="status-label">Status</p>
        <p className="status-value">{status}</p>
      </section>

      <section className="info-grid">
        <article className="card">
          <p className="card-label">Current Preset</p>
          <p className="card-value">{selectedPreset.key}. {selectedPreset.label}</p>
        </article>
        <article className="card">
          <p className="card-label">Playback</p>
          <p className="card-value">{isPlaying ? 'Playing' : 'Stopped'}</p>
        </article>
        <article className="card">
          <p className="card-label">Position</p>
          <p className="card-value">{formatTime(seekPosition)}</p>
        </article>
        <article className="card">
          <p className="card-label">Duration</p>
          <p className="card-value">{formatTime(durationSeconds)}</p>
        </article>
      </section>

      <section className="preset-list" aria-label="Preset keyboard map">
        {presets.length === 0 ? (
          <p>Loading presets...</p>
        ) : (
          presets.map((preset, index) => {
            const prepared = preparedPresets.includes(preset.name)
            const active = index === selectedPresetIndex
            return (
              <div
                key={preset.name}
                className={`preset-row${active ? ' active' : ''}${prepared ? '' : ' pending'}`}
              >
                <span className="preset-key">{String(index)}</span>
                <button
                  className="preset-button"
                  onClick={() => selectPreset(index)}
                  aria-label={`Select ${preset.label} preset (key ${index})`}
                  title={preset.expression}
                >
                  <span className="preset-name">{preset.label}</span>
                </button>
                <button
                  className="copy-button-small"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(preset.expression)
                      setStatus(`${preset.label} expression copied`)
                    } catch {
                      setStatus(`Failed to copy ${preset.label} expression`)
                    }
                  }}
                  aria-label={`Copy ${preset.label} expression to clipboard`}
                  title="Copy expression"
                >
                  Copy
                </button>
                <span className="preset-state">{prepared ? 'ready' : 'pending'}</span>
              </div>
            )
          })
        )}
      </section>

      <section className="expression-section" aria-label="Custom expression">
        <div className="expression-display">
          <p className="card-label">Current Expression</p>
          {currentExpression ? (
            <div className="expression-content">
              <code className="expression-text" aria-label="Current expression">
                {currentExpression}
              </code>
              <button
                className="copy-button"
                onClick={copyExpression}
                aria-label="Copy expression to clipboard"
                title="Copy to clipboard"
              >
                Copy
              </button>
            </div>
          ) : (
            <p className="no-expression">No custom expression applied</p>
          )}
        </div>

        <div className="expression-input-section">
          <label htmlFor="expression-input" className="card-label">
            Apply Custom Expression
          </label>
          <div className="expression-input-row">
            <input
              id="expression-input"
              type="text"
              className="expression-input"
              value={expressionInput}
              onChange={(e) => setExpressionInput(e.target.value)}
              placeholder="Enter effect chain expression (e.g., narrator | gain(2.0))"
              aria-describedby="expression-help"
              disabled={isSubmitting}
            />
            <button
              className="submit-button"
              onClick={submitExpression}
              disabled={!expressionInput.trim() || isSubmitting}
              aria-label="Apply expression to audio"
            >
              {isSubmitting ? 'Applying...' : 'Apply'}
            </button>
          </div>
          <p id="expression-help" className="expression-help">
            Enter an effect chain expression. The expression will be applied to the
            entire audio from the current playback position.
          </p>
        </div>
      </section>

      <section className="seek-section" aria-label="Playback controls">
        <label htmlFor="seek-slider" className="card-label">
          Seek
        </label>
        <div className="seek-controls">
          <input
            id="seek-slider"
            type="range"
            min="0"
            max={durationSeconds}
            step="0.1"
            value={seekPosition}
            onChange={handleSeek}
            onMouseUp={handleSeekCommit}
            onTouchEnd={handleSeekCommit}
            aria-label="Seek position"
            aria-valuemin={0}
            aria-valuemax={durationSeconds}
            aria-valuenow={seekPosition}
          />
          <span className="seek-time" aria-live="polite">
            {formatTime(seekPosition)} / {formatTime(durationSeconds)}
          </span>
        </div>
      </section>

      <audio ref={audioRef} preload="auto" />
    </main>
  )
}
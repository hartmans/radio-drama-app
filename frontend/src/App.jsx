import { useEffect, useRef, useState, useEffectEvent } from 'react'

const BACKEND_URL = (
  import.meta.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8000'
).replace(/\/+$/, '')
const PRESET_OPTIONS = [
  { key: '0', name: 'none', label: 'no preset' },
  { key: '1', name: 'narrator', label: 'narrator' },
  { key: '2', name: 'thoughts', label: 'thoughts' },
  { key: '3', name: 'outdoor1', label: 'outdoor1' },
  { key: '4', name: 'outdoor2', label: 'outdoor2' },
  { key: '5', name: 'indoor1', label: 'indoor1' },
  { key: '6', name: 'indoor2', label: 'indoor2' },
]

function formatTime(seconds) {
  const safeSeconds = Math.max(0, seconds)
  const wholeSeconds = Math.floor(safeSeconds)
  const minutes = Math.floor(wholeSeconds / 60)
  const remainder = wholeSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

export default function App() {
  const audioRef = useRef(null)
  const objectUrlRef = useRef(null)
  const [selectedPresetIndex, setSelectedPresetIndex] = useState(0)
  const [preparedPresets, setPreparedPresets] = useState([])
  const [playbackOffset, setPlaybackOffset] = useState(0)
  const [durationSeconds, setDurationSeconds] = useState(0)
  const [status, setStatus] = useState('Preparing presets')
  const [isPlaying, setIsPlaying] = useState(false)

  const selectedPreset = PRESET_OPTIONS[selectedPresetIndex]

  const releaseObjectUrl = useEffectEvent(() => {
    if (objectUrlRef.current !== null) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
  })

  const absolutePlaybackTime = useEffectEvent(() => {
    const audioElement = audioRef.current
    if (audioElement === null) {
      return playbackOffset
    }
    return playbackOffset + audioElement.currentTime
  })

  const fetchSlice = useEffectEvent(async (preset, fromTime, autoplay) => {
    setStatus(`Loading ${preset.label} from ${formatTime(fromTime)}`)
    const response = await fetch(`${BACKEND_URL}/api/audio-slice`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        preset_name: preset.name,
        from_time: fromTime,
      }),
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
    if (autoplay) {
      await audioElement.play()
      setIsPlaying(true)
      setStatus(`Playing ${preset.label} from ${formatTime(fromTime)}`)
      return
    }
    setIsPlaying(false)
    setStatus(`Ready on ${preset.label} at ${formatTime(fromTime)}`)
  })

  const startPlayback = useEffectEvent(async () => {
    const audioElement = audioRef.current
    if (audioElement !== null) {
      audioElement.pause()
    }
    await fetchSlice(selectedPreset, 0, true)
  })

  const stopPlayback = useEffectEvent(() => {
    const audioElement = audioRef.current
    if (audioElement === null) {
      return
    }
    const nextTime = absolutePlaybackTime()
    audioElement.pause()
    setPlaybackOffset(nextTime)
    setIsPlaying(false)
    setStatus(`Stopped at ${formatTime(nextTime)}`)
  })

  const selectPreset = useEffectEvent(async (presetIndex) => {
    const preset = PRESET_OPTIONS[presetIndex]
    const nextTime = absolutePlaybackTime()
    const shouldAutoplay = isPlaying
    const audioElement = audioRef.current
    if (audioElement !== null) {
      audioElement.pause()
    }
    setSelectedPresetIndex(presetIndex)
    await fetchSlice(preset, nextTime, shouldAutoplay)
  })

  const handleKeydown = useEffectEvent(async (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.repeat) {
      return
    }

    try {
      if (event.key === 'p' || event.key === 'P') {
        event.preventDefault()
        await startPlayback()
        return
      }
      if (event.key === 's' || event.key === 'S') {
        event.preventDefault()
        stopPlayback()
        return
      }
      if (event.key >= '0' && event.key <= '6') {
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
    if (audioElement === null) {
      return undefined
    }

    const handleEnded = () => {
      const nextTime = absolutePlaybackTime()
      setPlaybackOffset(nextTime)
      setIsPlaying(false)
      setStatus(`Finished at ${formatTime(nextTime)}`)
    }

    audioElement.addEventListener('ended', handleEnded)
    return () => {
      audioElement.removeEventListener('ended', handleEnded)
    }
  }, [])

  useEffect(() => {
    let active = true

    async function preparePresets() {
      try {
        const response = await fetch(`${BACKEND_URL}/api/presets/prepare`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            preset_names: PRESET_OPTIONS.map((preset) => preset.name),
          }),
        })
        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(errorText || `Backend request failed with ${response.status}`)
        }
        const payload = await response.json()
        if (!active) {
          return
        }
        setPreparedPresets(payload.preset_names)
        setDurationSeconds(payload.duration_seconds)
        setStatus(`Ready on ${selectedPreset.label}`)
      } catch (error) {
        if (!active) {
          return
        }
        setStatus(error instanceof Error ? error.message : String(error))
      }
    }

    preparePresets()

    const listener = (event) => {
      void handleKeydown(event)
    }
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

      <section className="status-panel" aria-live="polite">
        <p className="status-label">Status</p>
        <p className="status-value">{status}</p>
      </section>

      <section className="info-grid">
        <article className="card">
          <p className="card-label">Current Preset</p>
          <p className="card-value">
            {selectedPreset.key}. {selectedPreset.label}
          </p>
        </article>
        <article className="card">
          <p className="card-label">Playback</p>
          <p className="card-value">{isPlaying ? 'Playing' : 'Stopped'}</p>
        </article>
        <article className="card">
          <p className="card-label">Position</p>
          <p className="card-value">{formatTime(playbackOffset)}</p>
        </article>
        <article className="card">
          <p className="card-label">Duration</p>
          <p className="card-value">{formatTime(durationSeconds)}</p>
        </article>
      </section>

      <section className="preset-list" aria-label="Preset keyboard map">
        {PRESET_OPTIONS.map((preset, index) => {
          const prepared = preparedPresets.includes(preset.name)
          const active = index === selectedPresetIndex
          return (
            <div
              key={preset.name}
              className={`preset-row${active ? ' active' : ''}${prepared ? '' : ' pending'}`}
            >
              <span className="preset-key">{preset.key}</span>
              <span className="preset-name">{preset.label}</span>
              <span className="preset-state">{prepared ? 'ready' : 'pending'}</span>
            </div>
          )
        })}
      </section>

      <audio ref={audioRef} preload="auto" />
    </main>
  )
}

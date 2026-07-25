import { useEffect, useRef, useState, useCallback } from 'react'

// Base API URL - frontend constructs full URLs by appending filenames
const BACKEND_URL = 'http://10.36.0.202:8001/api'

function formatTime(seconds) {
  const safeSeconds = Math.max(0, seconds);
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

export default function App() {
  // Audio state - single source of truth for playback position  
  const audioRef = useRef(null);
  const [playbackOffset, setPlaybackOffset] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // Preset/expression state
  const [presetNames, setPresetNames] = useState([]);        // List of preset names
  const [presetExpressionsMap, setPresetExpressionsMap] = useState({});  // name -> expression
  const [selectedPresetIndex, setSelectedPresetIndex] = useState(0);
  const [currentExpression, setCurrentExpression] = useState('');
  const [expressionInput, setExpressionInput] = useState('');
  
  // Audio file tracking - we store just the filename, not full URL
  const [baseAudioFilename, setBaseAudioFilename] = useState(null);
  const [currentAudioFilename, setCurrentAudioFilename] = useState(null);
  
  // UI state  
  const [status, setStatus] = useState('Loading...');
  const [isApplying, setIsApplying] = useState(false);
  const [totalDuration, setTotalDuration] = useState(0);
  const [sampleRate, setSampleRate] = useState(48000);

  // Construct full URL from filename (no trailing slash handling needed)
  const constructFileUrl = useCallback((filename, options = {}) => {
    if (!filename) return null;
    
    // Ensure the filename has .wav extension if it doesn't already
    let cleanFilename = filename;
    if (!filename.endsWith('.wav')) {
      cleanFilename = `${filename}.wav`;
    }
    
    const { fromTime } = options;
    let url = `${BACKEND_URL}/cache/${cleanFilename}`;
    
    // Support seeking via query parameter - backend should honor Range requests anyway
    if (fromTime !== undefined && fromTime > 0) {
      url += `?from=${fromTime.toFixed(3)}`;
    }
    
    return url;
  }, []);

  // Load initial status: presets, base audio filename, duration
  useEffect(() => {
    async function loadStatus() {
      try {
        setStatus('Fetching production status...');
        
        const resp = await fetch(`${BACKEND_URL}/status`);
        if (!resp.ok) throw new Error(`Failed to fetch status: ${resp.status}`);
        
        const data = await resp.json();
        
        // Store preset data
        const names = Object.keys(data.preset_expressions).sort();
        setPresetNames(names);
        setPresetExpressionsMap(data.preset_expressions);
        
        if (names.length > 0) {
          setSelectedPresetIndex(0);
          setExpressionInput(data.preset_expressions[names[0]] || '');
          
          // Select the "narrator" preset by default if available, otherwise first one
          const narratorIdx = names.indexOf('narrator');
          if (narratorIdx >= 0) setSelectedPresetIndex(narratorIdx);
        }
        
        setBaseAudioFilename(data.base_audio_file);
        setTotalDuration(data.total_duration_seconds);
        setSampleRate(data.sample_rate);
        
        // Load base audio immediately - construct URL from filename only
        const baseUrl = constructFileUrl(data.base_audio_filename);
        if (baseUrl) {
          try {
            const sliceResp = await fetch(baseUrl);
            if (sliceResp.ok) {
              const blob = await sliceResp.blob();
              const audio = audioRef.current;
              if (audio) {
                audio.src = URL.createObjectURL(blob);
                audio.load();
                setCurrentAudioFilename(data.base_audio_filename);
              }
            } else {
              console.warn('Base audio fetch failed, will load on demand');
            }
          } catch (e) {
            console.warn('Base audio load error:', e.message);
          }
        }
        
        setStatus('Ready - select preset and press Play');
      } catch (err) {
        console.error('Failed to load status:', err);
        setStatus(err.message || String(err));
      }
    }
    
    loadStatus();
  }, [constructFileUrl]);

  // Apply expression handler
  const applyExpression = useCallback(async (expression, fromTime = 0) => {
    if (!expression.trim()) return;
    
    setIsApplying(true);
    setStatus('Applying expression...');
    
    try {
      const response = await fetch(`${BACKEND_URL}/apply-expression`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({expression, from_time: fromTime}),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error ${response.status}`);
      }
      
      const data = await response.json();
      
      // Update total duration and sample rate from response
      if (data.duration_seconds !== undefined) setTotalDuration(data.duration_seconds);
      if (data.sample_rate !== undefined) setSampleRate(data.sample_rate);
      
      // Determine if we were playing before applying
      const wasPlaying = isPlaying;
      const playPosition = audioRef.current?.currentTime || 0;
      
      // Load the new audio from cache - construct URL from returned filename only
      const newAudioUrl = constructFileUrl(data.filename);
      if (!newAudioUrl) throw new Error('No filename in response');
      
      const blobResp = await fetch(newAudioUrl);
      if (!blobResp.ok) throw new Error(`Failed to load cached audio: ${blobResp.status}`);
      
      const blob = await blobResp.blob();
      const audio = audioRef.current;
      if (audio) {
        // Release old object URL if exists
        if (currentAudioFilename && currentAudioFilename !== baseAudioFilename) {
          // Note: we're creating new object URLs each time, browser handles cleanup
        }
        
        audio.src = URL.createObjectURL(blob);
        audio.load();
        
        // Seek to appropriate position
        if (fromTime > 0 && fromTime < data.duration_seconds) {
          audio.currentTime = fromTime;
        } else if (wasPlaying) {
          // Preserve playback position if we were playing
          audio.currentTime = Math.min(playPosition, data.duration_seconds);
          // Resume playback automatically
          audio.play()
            .then(() => setIsPlaying(true))
            .catch(e => console.warn('Auto-play failed:', e.message));
          setStatus(`Applied: ${expression.split(' | ')[0]?.slice(0, 40)}... (resuming)`);
        } else {
          audio.currentTime = 0;
          setStatus(`Applied: ${expression.split(' | ')[0]?.slice(0, 40)}`);
        }
      }
      
      setCurrentExpression(expression);
      setCurrentAudioFilename(data.filename);
      setExpressionInput(expression);
      setIsApplying(false);
    } catch (err) {
      setStatus(`Failed to apply expression: ${err.message}`);
      setIsApplying(false);
    }
  }, [BACKEND_URL, constructFileUrl, currentAudioFilename, baseAudioFilename, isPlaying]);

  const submitExpression = useCallback(() => {
    if (!expressionInput.trim() || isApplying) return;
    
    // Get current playback position as from_time
    const audio = audioRef.current;
    const currentTime = audio ? Math.max(0, audio.currentTime) : 0;
    
    applyExpression(expressionInput.trim(), currentTime);
  }, [expressionInput, isApplying, applyExpression]);

  // Keyboard shortcuts - runs after presets are loaded  
  useEffect(() => {
    if (presetNames.length === 0) return;
    
    const handler = (e) => {
      if (e.altKey || e.ctrlKey || e.metaKey || e.repeat) return;
      
      const isInputFocused = document.activeElement.tagName === 'INPUT' && 
                             document.activeElement.type !== 'submit';
      
      // Number keys select preset (only when NOT in input field)  
      if (!isInputFocused && e.key >= '0' && e.key < String(presetNames.length)) {
        e.preventDefault();
        const idx = parseInt(e.key, 10);
        setSelectedPresetIndex(idx);
        const expr = presetExpressionsMap[presetNames[idx]] || '';
        setExpressionInput(expr);
        setCurrentExpression(expr);
        setStatus(`Loaded preset: ${presetNames[idx]}. Press Apply to render.`);
      }

      // Space toggles playback (only when NOT in input field)  
      if (!isInputFocused && e.key === ' ') {
        e.preventDefault();
        const audio = audioRef.current;
        if (!audio || !audio.src) return;
        
        if (isPlaying) {
          audio.pause();
          setIsPlaying(false);
        } else {
          audio.play().then(() => setIsPlaying(true)).catch(console.warn);
        }
      }
    };
    
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [presetNames, presetExpressionsMap, isPlaying]);

  // Audio event handlers  
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    
    const onLoadedMeta = () => {
      setTotalDuration(audio.duration || totalDuration);
    };
    
    const onUpdate = () => {
      setPlaybackOffset(audio.currentTime);
    };
    
    const onEnd = () => setIsPlaying(false);
    
    audio.addEventListener('loadedmetadata', onLoadedMeta);
    audio.addEventListener('timeupdate', onUpdate);  
    audio.addEventListener('ended', onEnd);
    
    return () => {
      audio.removeEventListener('loadedmetadata', onLoadedMeta);
      audio.removeEventListener('timeupdate', onUpdate);
      audio.removeEventListener('ended', onEnd);
    };
  }, [totalDuration]);

  // Copy expression handler
  const copyExpression = useCallback(async () => {
    const textToCopy = expressionInput || currentExpression;
    if (!textToCopy) return;
    
    try {
      await navigator.clipboard.writeText(textToCopy);
      setStatus('Copied to clipboard!');
    } catch (err) {
      setStatus(`Copy failed: ${err.message}`);
    }
  }, [expressionInput, currentExpression]);

  return (
    <main className="app-shell" aria-label="Effect Chain Editor">
      <header className="hero">
        <p className="eyebrow">EFFECT CHAIN EDITOR</p>  
        <h1>Preset Manager</h1>
        <p className="summary">Select preset to load expression, type custom chain, or edit. Press Apply to render and cache.</p>
      </header>

      <section className="status-panel" aria-live="polite">{`Status: ${status || 'Ready'}`}</section>

      {/* Hidden audio element - controls via keyboard (Space) */}
      <audio 
        ref={audioRef} 
        preload="auto" 
        onError={(e) => setStatus(`Audio error: ${e.target.error?.message || '?'}`)}
      />

      {/* Visual playback info */}
      {currentAudioFilename && (
        <section className="now-playing">
          <small>{`Playing: ${currentAudioFilename.slice(0, 16)}... (${sampleRate} Hz)`}</small>
        </section>
      )}

      <section className="controls" style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
        {/* Play/Pause button */}
        <button 
          onClick={() => {
            const audio = audioRef.current;
            if (!audio || !audio.src) return;
            
            if (isPlaying) {
              audio.pause();
              setIsPlaying(false);
            } else {
              audio.play().then(() => setIsPlaying(true)).catch(e => setStatus(`Play error: ${e.message}`));
            }
          }}
          disabled={!currentAudioFilename}
          style={{ 
            minWidth: '60px',
            padding: '0.5rem 1rem',
            cursor: currentAudioFilename ? 'pointer' : 'not-allowed',
            backgroundColor: isPlaying ? '#ef4444' : '#22c55e',
            color: 'white',
            border: 'none',
            borderRadius: '4px'
          }}
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>

        {/* Seek slider */}
        {totalDuration > 0 && (
          <div className="seek-controls" style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>{formatTime(playbackOffset)}</span>
            <input 
              type="range" 
              min="0" 
              max={totalDuration} 
              step="0.1"
              value={playbackOffset} 
              onChange={(e) => setPlaybackOffset(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <span>{formatTime(totalDuration)}</span>
          </div>
        )}
        
        {/* Apply button for expression */}
        {isApplying && <span className="apply-status">Rendering...</span>}
      </section>

      {/* Expression input section - always show base audio URL hint when no custom expression applied */}
      <section className="expression-section" style={{ 
        padding: '1rem', 
        border: currentAudioFilename === baseAudioFilename ? '1px solid #4caf50' : '1px solid #333',
        borderRadius: '4px',
        marginBottom: '1rem'
      }}>
        <h2>Current Audio</h2>
        {currentAudioFilename === baseAudioFilename && (
          <div style={{ color: '#888', marginBottom: '0.5rem' }}>
            No expression applied yet - playing base audio from cache.<br/>
            Path fragment: /api/cache/{baseAudioFilename}
          </div>
        )}
      </section>

      <section className="expression-section" aria-label="Expression Editor">
        <label htmlFor="expression">Effect Chain Expression</label>
        <input 
          id="expression" 
          type="text" 
          value={expressionInput} 
          onChange={(e) => setExpressionInput(e.target.value)} 
          placeholder='filter_audio(btype="highpass", cutoff_hz=100.0) | compress_audio(...)' 
          style={{
            width: '100%',
            maxWidth: '80ch',
            fontFamily: 'monospace',
            fontSize: '0.9rem'
          }}
        />
        
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
          <button 
            onClick={copyExpression} 
            disabled={!expressionInput.trim()} 
            aria-label="Copy to clipboard"
            style={{ minWidth: '8ch' }}
          >
            COPY
          </button>  
          
          <button 
            onClick={submitExpression} 
            disabled={!expressionInput.trim() || isApplying}
            style={{ minWidth: '10ch' }}
          >
            {isApplying ? 'APPLYING...' : 'APPLY'}
          </button>
        </div>
      </section>

      {/* Preset selection section */}
      <section className="preset-section" aria-label="Presets">
        <h2>Presets (number keys 0-9 to select)</h2>
        {presetNames.length === 0 ? (
          <p>Loading presets...</p>
        ) : (
          presetNames.map((name, i) => (
            <button 
              key={name} 
              onClick={() => { 
                setSelectedPresetIndex(i); 
                const expr = presetExpressionsMap[name] || '';
                setExpressionInput(expr);
                setCurrentExpression(expr);
                setStatus(`Loaded preset: ${name}. Press Apply to render.`);
              }}
              style={{
                fontWeight: i === selectedPresetIndex ? 'bold' : 'normal',
                backgroundColor: i === selectedPresetIndex ? '#4caf50' : '',
                color: i === selectedPresetIndex ? 'white' : ''
              }}
            >
              {i}: {name}... 
            </button>
          ))
        )}
      </section>

      {/* Expression display in text form */}
      <section className="expression-view">
        <h3>Expression: {currentExpression.slice(0, 120)}{currentExpression.length > 120 ? '...' : ''}</h3>
        <small>(Full expression shown above in input)</small>
      </section>
    </main>
  );
}

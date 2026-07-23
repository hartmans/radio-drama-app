import { useEffect, useRef, useState, useCallback } from 'react'

const BACKEND_URL = 'http://10.36.0.202:8001'  // Direct connection to backend on same machine

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
  
  // Expression/preset state
  const [presets, setPresets] = useState([]);
  const [selectedPresetIndex, setSelectedPresetIndex] = useState(0);
  const [currentExpression, setCurrentExpression] = useState('');
  const [expressionInput, setExpressionInput] = useState('');
  
  // UI state  
  const [status, setStatus] = useState('Loading presets...');
  const [isApplying, setIsApplying] = useState(false);
  const [durationSeconds, setDurationSeconds] = useState(1486);

  // Helper: applyExpression (defined here so keyboard handler can reference it)  
  const applyExpressionRaw = useCallback(async (expression, fromTime) => {
    setStatus('Applying expression...');
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/apply-expression`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({expression, from_time: fromTime}),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error ${response.status}`);
      }
      
      const data = await response.json();
      if (data.duration_seconds !== undefined) setDurationSeconds(data.duration_seconds);
      setPlaybackOffset(fromTime);  
      setCurrentExpression(expression);
      setIsApplying(false);
      setStatus(`Applied: ${expression.split(' | ')[0]?.slice(0, 40)}...`);
    } catch (err) {
      setStatus(`Failed to apply expression: ${err.message}`);
      setIsApplying(false);
    }
  }, []);

  const submitExpression = useCallback(() => {
    if (!expressionInput.trim() || isApplying) return;
    setIsApplying(true);
    const audio = audioRef.current;
    const currentTime = audio ? playbackOffset + (audio.currentTime - playbackOffset) : playbackOffset;
    applyExpressionRaw(expressionInput.trim(), currentTime);
  }, [expressionInput, isApplying, playbackOffset, applyExpressionRaw]);

  // Load preset expressions and base audio on mount  
  useEffect(() => {
    async function loadAll() {
      try {
        const resp = await fetch(`${BACKEND_URL}/api/presets/expressions`);
        if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status}`);
        
        const exprMap = await resp.json();
        const presetList = Object.entries(exprMap)
          .filter(([name]) => name !== 'master')
          .map(([name, expression]) => ({
            name,
            label: name.replace(/([A-Z])/g, ' $1').trim(),
            expression,  
          }));
        
        setPresets(presetList);
        setStatus('Ready - select preset and press Play');
        
        // Load base audio immediately so Play works
        const audio = audioRef.current;
        if (audio && presetList.length > 0) {
          try {
            const sliceResp = await fetch(`${BACKEND_URL}/api/audio-slice`, {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({preset_name: 'none', from_time: 0}),
            });
            if (!sliceResp.ok) throw new Error(`Audio load failed: ${sliceResp.status}`);
            
            const blob = await sliceResp.blob();
            audio.src = URL.createObjectURL(blob);
            audio.load();
          } catch (err) {
            setStatus(`Base audio error: ${err.message}`);
          }
        }
      } catch (err) {
        setStatus(err.message || String(err));
      }
    }
    
    loadAll();
  }, []);

  // Keyboard shortcuts - runs after presets are loaded  
  useEffect(() => {
    const handler = (e) => {
      if (e.altKey || e.ctrlKey || e.metaKey || e.repeat) return;
      
      const isInputFocused = document.activeElement.id === 'expression';
      
      // Number keys select preset (only when NOT in input field)  
      if (!isInputFocused && e.key >= '0' && e.key < String(presets.length)) {
        e.preventDefault();
        setSelectedPresetIndex(parseInt(e.key, 10));
        setExpressionInput(presets[parseInt(e.key, 10)].expression);
        setStatus(`Loaded: ${presets[parseInt(e.key, 10)].label}. Click Apply to use.`);
      }

      // P toggles playback (only when NOT in input field)  
      if (!isInputFocused && (e.key === 'p' || e.key === 'P')) {
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
  }, [presets, isPlaying]);

  // Audio event handlers  
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    
    const onLoadedMeta = () => setDurationSeconds(audio.duration || 1486);
    const onUpdate = () => setPlaybackOffset(audio.currentTime);
    const onEnd = () => setIsPlaying(false);
    
    audio.addEventListener('loadedmetadata', onLoadedMeta);
    audio.addEventListener('timeupdate', onUpdate);  
    audio.addEventListener('ended', onEnd);
    
    return () => {
      audio.removeEventListener('loadedmetadata', onLoadedMeta);
      audio.removeEventListener('timeupdate', onUpdate);
      audio.removeEventListener('ended', onEnd);
    };
  }, []);

  // Copy expression  
  const copyExpression = useCallback(async () => {
    const textToCopy = presets[selectedPresetIndex]?.expression || expressionInput;
    if (!textToCopy) return;
    try {
      await navigator.clipboard.writeText(textToCopy);
      setStatus('Copied!');
    } catch (err) {
      setStatus(`Copy failed: ${err.message}`);
    }
  }, [presets, selectedPresetIndex, expressionInput]);

  return (
    <main className="app-shell" aria-label="Effect Chain Editor">
      <header className="hero">
        <p className="eyebrow">EFFECT CHAIN EDITOR</p>  
        <h1>Preset Manager</h1>
        <p className="summary">Click preset to load expression, or type custom chain. Press Apply to re-render audio from current position.</p>
      </header>

      <section className="status-panel" aria-live="polite">{`Status: ${status || 'Ready'}`}</section>

      <audio ref={audioRef} preload="auto" onError={(e) => setStatus(`Audio error: ${e.target.error?.message || '?'}`)} />

      <section className="controls">
        <button onClick={() => { /* toggle handled by keyboard or direct play() */ }} disabled>
          ▶ Play (use P key or click audio element)
        </button>
        
        <div className="seek-controls">
          <input type="range" min="0" max={durationSeconds} value={playbackOffset} onChange={(e)=>setPlaybackOffset(parseFloat(e.target.value))}/>
          <span>{formatTime(playbackOffset)} / {formatTime(durationSeconds)}</span>
        </div>
      </section>

      <section className="expression-section">
        <label htmlFor="expression">Effect Chain</label>
        <input id="expression" type="text" value={expressionInput} onChange={(e)=>setExpressionInput(e.target.value)} placeholder="filter_audio(...) | compress_audio(...)" />
        <div style={{display:'flex',gap:'0.5rem'}}>
          <button onClick={copyExpression} disabled={!expressionInput.trim()} aria-label="Copy">COPY</button>  
          <button onClick={submitExpression} disabled={!expressionInput.trim() || isApplying}>{isApplying?'APPLYING...':'APPLY'}</button>
        </div>
      </section>

      <section className="preset-section" aria-label="Presets">
        <h2>Presets (click to load)</h2>
        {presets.length === 0 ? <p>Loading...</p> : presets.map((p, i) => (
          <button key={p.name} onClick={() => { 
            setSelectedPresetIndex(i); 
            setExpressionInput(p.expression);  
            setStatus(`Loaded: ${p.label}. Click Apply.`);
          }}>
            {i}: {p.label.split(' ')[0]}... 
          </button>
        ))}
      </section>
    </main>
  );
}

# VoxCPM2 engine

The engine renders lines serially.  It always supplies the original speaker
sample as VoxCPM2's `reference_wav_path`, preserving the speaker's identity.

For a line that begins with a non-empty parenthetical, such as
`(flustered, quiet)I understand.`, the parenthetical is a VoxCPM2 control
instruction.  The line uses reference-only cloning: it does not supply a
prompt WAV or prompt text, because VoxCPM2's continuation/ultimate-cloning
mode does not support style controls.

For every other line, the engine uses continuation cloning.  The first such
line for a speaker is prompted with that speaker's original reference WAV and
its reference transcript. A controlled line becomes the continuation prompt
for later un-controlled lines from that speaker; its parenthetical control is
excluded from the stored prompt text. The engine keeps that prompt fixed rather
than chaining each generated line into the next, avoiding progressive voice
degradation.

Prompt state is scoped to one render request, so concurrently submitted script
requests do not condition one another.

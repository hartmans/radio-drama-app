"""Protocol demonstration engine that emits silence without model dependencies."""

from pathlib import Path

from radio_drama_tts_container import artifact_name, run_server, write_pcm16_wav


SAMPLE_RATE = 24_000


def render_batch(requests):
    results = []
    for request in requests:
        name = artifact_name(request)
        line_count = sum(
            item.get("type") == "line" for item in request["dialogue_contents"]
        )
        frame_count = max(1, line_count) * SAMPLE_RATE // 4
        write_pcm16_wav(Path(name), (0.0 for _ in range(frame_count)), sample_rate=SAMPLE_RATE)
        results.append({"wav": name})
    return results


if __name__ == "__main__":
    run_server(render_batch)

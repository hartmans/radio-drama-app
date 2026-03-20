from __future__ import annotations

import pytest

from radio_drama.document import parse_production_string
from radio_drama.errors import DocumentError


def test_parse_production_builds_phase1_tree():
    root = parse_production_string(
        """
        <production>
          <speaker-map>
            anna: anna.wav
            ben: ben.wav
          </speaker-map>
          <script>
            Anna: First line.
            Ben: Response.
          </script>
        </production>
        """,
        source_name="example.xml",
    )

    assert root.tag_name == "production"
    assert root.location.source == "example.xml"
    assert root.speaker_map_node.normalized_text_content == "anna: anna.wav\nben: ben.wav"
    assert len(root.script_nodes) == 1
    assert "Anna: First line." in root.script_nodes[0].normalized_text_content


def test_parse_production_collects_element_attributes():
    root = parse_production_string(
        """
        <production demo="presets">
          <speaker-map source="inline">anna: anna.wav</speaker-map>
          <script preset="narrator1" mood="internal">Anna: First line.</script>
        </production>
        """,
        source_name="attrs.xml",
    )

    assert root.attributes == {"demo": "presets"}
    assert root.speaker_map_node.attributes == {"source": "inline"}
    assert root.script_nodes[0].attributes == {"preset": "narrator1", "mood": "internal"}
    assert root.script_nodes[0].preset == "narrator1"


def test_parse_production_rejects_unknown_child():
    with pytest.raises(DocumentError, match="does not allow child element <scene>"):
        parse_production_string(
            """
            <production>
              <scene />
            </production>
            """,
            source_name="bad.xml",
        )


def test_parse_production_wraps_xml_parse_errors():
    with pytest.raises(DocumentError, match="mismatched tag"):
        parse_production_string("<production><script></production>", source_name="broken.xml")

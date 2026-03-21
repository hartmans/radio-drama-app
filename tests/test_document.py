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


def test_script_allows_sound_nodes_by_attribute_or_text():
    root = parse_production_string(
        """
        <production>
          <speaker-map>anna: anna.wav</speaker-map>
          <script>
            Anna: Open the door.
            <sound ref="door" />
            <sound>footsteps</sound>
          </script>
        </production>
        """,
        source_name="sound.xml",
    )

    sound_nodes = root.script_nodes[0].child_elements_named("sound")
    assert len(sound_nodes) == 2
    assert sound_nodes[0].ref == "door"
    assert sound_nodes[1].ref == "footsteps"


def test_production_allows_direct_sound_without_speaker_map():
    root = parse_production_string(
        """
        <production>
          <sound ref="door" />
        </production>
        """,
        source_name="production-sound.xml",
    )

    sound_nodes = root.child_elements_named("sound")
    assert len(sound_nodes) == 1
    assert sound_nodes[0].ref == "door"


def test_script_allows_nested_script():
    root = parse_production_string(
        """
        <production>
          <speaker-map>anna: anna.wav</speaker-map>
          <script>
            Anna: Outer line.
            <script>Anna: Inner line.</script>
          </script>
        </production>
        """,
        source_name="nested-script.xml",
    )

    nested_scripts = root.script_nodes[0].child_elements_named("script")
    assert len(nested_scripts) == 1
    assert "Anna: Inner line." in nested_scripts[0].normalized_text_content


def test_sound_rejects_missing_ref():
    with pytest.raises(DocumentError, match="<sound> requires either a ref attribute or text content"):
        parse_production_string(
            """
            <production>
              <speaker-map>anna: anna.wav</speaker-map>
              <script><sound /></script>
            </production>
            """,
            source_name="missing-sound.xml",
        )


def test_sound_rejects_mismatched_text_and_ref():
    with pytest.raises(
        DocumentError,
        match="<sound> text content must match the ref attribute when both are present",
    ):
        parse_production_string(
            """
            <production>
              <speaker-map>anna: anna.wav</speaker-map>
              <script><sound ref="door">window</sound></script>
            </production>
            """,
            source_name="mismatch-sound.xml",
        )


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

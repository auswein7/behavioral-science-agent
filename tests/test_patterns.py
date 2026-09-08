"""Unit tests for the shared leak-detection patterns, focused on likely_name
precision: sentence-initial words after Markdown boundaries are not names."""

from src.scrub.patterns import iter_findings


def likely_names(text: str) -> list[str]:
    return [term for cat, term, _ in iter_findings(text) if cat == "likely_name"]


class TestLikelyNamePrecision:
    def test_midsentence_name_flagged(self):
        assert likely_names("ask Sarah about the case") == ["Sarah"]

    def test_sentence_initial_word_excluded(self):
        assert likely_names("the group pauses. Then everyone nods.") == []

    def test_dialogue_marker_is_a_boundary(self):
        # Screenplay dialogue lines put "> " mid-line after the speaker tag;
        # the word right after it is sentence-initial, not a name.
        assert likely_names("**SPEAKER_04** *(00:00:03.656)* > Even so, fine.") == []

    def test_name_after_dialogue_marker_still_flagged_midsentence(self):
        assert "David" in likely_names("> They said hello to David yesterday")

    def test_markdown_heading_start_excluded(self):
        assert likely_names("# Scene 1\nSPEAKER_01 looks left.") == []

    def test_line_start_after_plain_newline_excluded(self):
        assert likely_names("first line ends here\nThen a new line starts") == []

    def test_first_person_pronoun_allowlisted(self):
        assert likely_names("and I think I said so") == []

    def test_colon_is_a_boundary(self):
        assert likely_names("the note reads: Remember the meeting") == []

    def test_speaker_tags_never_flagged(self):
        assert likely_names("and SPEAKER_02 turns toward SPEAKER_03 fast") == []

    def test_scrub_tokens_never_flagged(self):
        assert likely_names("thanks [NAME], see you at [PLACE] then") == []


class TestFirstLineIsALineStart:
    """A document opening with a Markdown heading got no line-start treatment,
    because \\A did not allow the prefix that \\n did. The canonical title
    tripped likely_name on line 1 and would not have on line 2."""

    @staticmethod
    def _names(text: str) -> list[str]:
        return [term for category, term, _ in iter_findings(text) if category == "likely_name"]

    def test_canonical_title_on_the_first_line_is_clean(self):
        assert self._names("# Screenplay\n\n*A screenplay.*\n") == []

    def test_same_heading_on_a_later_line_agrees(self):
        first = self._names("# Screenplay\n")
        later = self._names("intro text.\n\n# Screenplay\n")
        assert first == later == []

    def test_a_real_name_in_a_heading_is_still_caught(self):
        # The whole reason to fix the boundary rather than skip headings.
        assert "Sarah" in self._names("# Report by Sarah Connor\n")

    def test_generated_title_still_flags_its_mid_title_capitals(self):
        # Run 4's title. Only the first word gets line-start treatment; the
        # rest are real mid-line capitals and the gate should say so.
        names = self._names("# A Responsibility for Our Blue Planet\n")
        assert names == ["Responsibility", "Our", "Blue", "Planet"]

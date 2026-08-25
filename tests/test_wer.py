"""WER math (validated against a hand-worked example), normalization, loop
detection, and channel mixing. No live Whisper here."""

import numpy as np

from s2n.transcription.wer import (
    detect_repetition_loop,
    normalize_primock,
    word_error_rate,
)


# ---- WER: hand-worked example --------------------------------------------
def test_wer_hand_worked_example():
    """ref: 'the cat sat on the mat'  (6 words)
    hyp: 'cat sit on the mat quickly'
      leading 'the' gone = 1 deletion
      sat->sit           = 1 substitution
      'quickly' added    = 1 insertion
    WER = (1+1+1)/6 = 0.5 exactly. (Chosen so the optimal alignment is UNIQUE —
    examples with repeated words can have several equal-cost alignments, which
    makes the S/D/I split ambiguous even though the WER itself is fixed.)"""
    r = word_error_rate("the cat sat on the mat", "cat sit on the mat quickly")
    assert r["wer"] == 0.5
    assert r["substitutions"] == 1 and r["deletions"] == 1 and r["insertions"] == 1
    assert r["n_ref_words"] == 6


def test_wer_identity_and_extremes():
    assert word_error_rate("a b c", "a b c")["wer"] == 0.0
    assert word_error_rate("a b c", "")["wer"] == 1.0  # all deleted
    r = word_error_rate("", "a b")  # empty reference -> undefined (NaN)
    assert r["wer"] != r["wer"] and r["insertions"] == 2


def test_wer_can_exceed_one():
    assert word_error_rate("a", "x y z")["wer"] == 3.0  # 1 sub + 2 ins over 1 ref word


# ---- normalization ---------------------------------------------------------
def test_primock_normalization():
    assert normalize_primock("Well - I've got, um... a COUGH!") == "well i've got um a cough"


# ---- repetition-loop detector ----------------------------------------------
def test_loop_detector_flags_unigram_and_bigram_loops():
    assert detect_repetition_loop("bye " * 10).looping
    f = detect_repetition_loop("ok so " * 6 + "then I left")  # bigram, any offset
    assert f.looping and f.ngram == "ok so" and f.repeats >= 5


def test_loop_detector_allows_normal_speech():
    assert not detect_repetition_loop("yeah yeah I know I know it hurts a lot").looping
    assert not detect_repetition_loop("no no no okay").looping  # 3x < threshold 5


# ---- channel mixing ---------------------------------------------------------
def test_mix_channels_averages_samples(tmp_path):
    import wave

    from s2n.transcription.whisper_asr import mix_channels

    def write(path, values):
        with wave.open(str(path), "wb") as w:
            w.setparams((1, 2, 16000, 0, "NONE", "no compression"))
            w.writeframes(np.array(values, dtype=np.int16).tobytes())

    write(tmp_path / "d.wav", [1000, -2000, 0])
    write(tmp_path / "p.wav", [3000, 2000, 0, 500])  # longer -> zero-pad the other
    out = mix_channels(tmp_path / "d.wav", tmp_path / "p.wav", tmp_path / "m.wav")

    with wave.open(str(out), "rb") as w:
        mixed = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert list(mixed) == [2000, 0, 0, 250]  # element-wise average, padded

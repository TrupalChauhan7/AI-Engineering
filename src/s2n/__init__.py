"""s2n — Speech-to-Clinical-Note.

A pipeline that turns consultation audio into a SOAP note, then measures
how faithful that note is and raises a reliability flag on unreliable ones.

Sub-packages:
    data          load PriMock57 (audio, transcripts, notes, human eval)
    transcription Whisper wrapper                              (RQ2)
    generation    transcript + prompt -> SOAP note (LLM)
    evaluation    metrics + LLM-judge + highlights + correlation  (the heart)
    reliability         the reliability flagger                       (the star)
    llm           backend-agnostic LLM client (swap freely)
    utils         shared helpers
"""

__version__ = "0.1.0"

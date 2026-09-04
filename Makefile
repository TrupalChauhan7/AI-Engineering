# One obvious command per task. Run e.g. `make transcribe`.
.PHONY: install transcribe generate evaluate test demo samples finetune-train

install:      ## install the s2n package (editable) into the active env
	pip install -e .

transcribe:   ## RQ2: run Whisper over the audio
	python pipelines/01_transcribe.py

generate:     ## generate SOAP notes from transcripts
	python pipelines/02_generate_notes.py

evaluate:     ## score notes + run the reliability flag
	python pipelines/03_evaluate.py

test:         ## run unit tests
	pytest -q

demo:         ## run the Clarion demo (FastAPI :8000 + Next.js :3000)
	python scripts/dev.py

samples:      ## pre-compute the demo sample consultations (gitignored output)
	python scripts/build_samples.py

finetune-train:   ## reproduce the Phase-15b LoRA adapter (Track B). SLOW — hours, ~14.4 GB.
	@echo "Reproduces results/finetune/best_adapter. Use DRY=1 to just print the command."
	python pipelines/12_finetune_train.py $(if $(DRY),--dry-run,)

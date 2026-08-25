# Demo app (the "stage")

Small front-end for the demo: upload consultation audio → see the generated
SOAP note + the reliability flag.

Planned (build AFTER the core pipeline works): a Streamlit app (`app.py`) that
calls the s2n library. Kept separate so demo code never mixes with research code.

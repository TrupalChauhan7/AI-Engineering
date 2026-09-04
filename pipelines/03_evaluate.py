"""Pipeline step 3 — score notes, run the reliability, validate vs humans (RQ1).

Run:  python pipelines/03_evaluate.py   (or: make evaluate)
"""

from s2n.config import load_config


def main():
    cfg = load_config()
    print(
        "TODO: metrics + judge + reliability @ bands",
        cfg["reliability"]["reliable_max"],
        cfg["reliability"]["unreliable_min"],
    )


if __name__ == "__main__":
    main()

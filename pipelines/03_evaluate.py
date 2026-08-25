"""Pipeline step 3 — score notes, run the alarm, validate vs humans (RQ1).

Run:  python pipelines/03_evaluate.py   (or: make evaluate)
"""

from s2n.config import load_config


def main():
    cfg = load_config()
    print(
        "TODO: metrics + judge + alarm @ bands",
        cfg["alarm"]["reliable_max"],
        cfg["alarm"]["unreliable_min"],
    )


if __name__ == "__main__":
    main()

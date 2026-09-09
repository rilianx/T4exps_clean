#!/usr/bin/env python3
"""Compara los brazos de calibracion con orden aleatorio.

Los exitos NO son iid: cada replica reporta su propia likelihood, asi que el
test correcto es una Poisson-binomial con p_i = L_i.  Como la L es una cota
inferior de P(correcto), sum(L_i) son los aciertos ESPERADOS y una cola
izquierda chica (P(X <= observados)) es evidencia de sobreconfianza.
"""
import glob, json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
REF = [4.0, 2.0, 0.4, 0.01]

ARMS = [                                   # (etiqueta, patron, nivel)
    ("prefix / mean", "stage3_replay_calr_solver_o{o}_c{c}_result.json", "0.8"),
    ("auto / mean",   "stage3_replay_d5_mean_o{o}_c{c}_result.json",     "0.8"),
    ("auto / best",   "stage3_replay_d5_best_o{o}_c{c}_result.json",     "0.8"),
    ("prefix / mean", "stage3_replay_calr_solver_o{o}_c{c}_result.json", "0.98"),
    ("auto / mean",   "stage3_calauto_o{o}_c{c}_result.json",            "0.98"),
]


def poisson_binomial_cdf(k, ps):
    dist = [1.0]
    for p in ps:
        nd = [0.0] * (len(dist) + 1)
        for i, v in enumerate(dist):
            nd[i] += v * (1 - p)
            nd[i + 1] += v * p
        dist = nd
    return sum(dist[: k + 1])


def main():
    out = []
    for label, pat, lev in ARMS:
        rows = []
        for o in range(1, 33):
            f = os.path.join(HERE, pat.format(o=o, c=lev))
            if os.path.exists(f):
                d = json.load(open(f))
                rows.append(dict(order=o, runs=d["runs"], L=d["likelihood"],
                                 correct=d["output"] == REF, output=d["output"],
                                 misses=d.get("misses", 0)))
        if not rows:
            continue
        ok = sum(r["correct"] for r in rows); ps = [r["L"] for r in rows]
        out.append(dict(arm=label, level=float(lev), n=len(rows), correct=ok,
                        expected=sum(ps), p_value=poisson_binomial_cdf(ok, ps),
                        runs=sum(r["runs"] for r in rows) / len(rows),
                        reals=sum(r["misses"] for r in rows) / len(rows),
                        wrong=[f"o{r['order']}" for r in rows if not r["correct"]],
                        rows=rows))
    json.dump(dict(ref=REF, arms=out), open(os.path.join(HERE, "calibration_arms.json"), "w"), indent=1)
    print(f"{'nivel':>6} {'brazo':<15} {'n':>3} {'ok':>3} {'esperados':>10} {'P(X<=ok)':>9} {'runs':>6}  incorrectas")
    for a in sorted(out, key=lambda x: (x["level"], x["arm"])):
        flag = "  <- sobreconfianza" if a["p_value"] < 0.05 else ""
        print(f"{a['level']:>6} {a['arm']:<15} {a['n']:>3} {a['correct']:>3} {a['expected']:>10.2f} "
              f"{a['p_value']:>9.3f} {a['runs']:>6.0f}  {', '.join(a['wrong'])}{flag}")


if __name__ == "__main__":
    main()

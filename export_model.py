"""Export svm_model.pkl + tfidf_vectorizer.pkl to a plain JSON file (api/model.json).

The JSON file lets the Vercel function predict without scikit-learn.
Run once on your computer:  python export_model.py
"""
import json
import joblib
import numpy as np

tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("svm_model.pkl")

R = 7  # decimals kept
folds = []
for c in model.calibrated_classifiers_:
    est = getattr(c, "estimator", None) or getattr(c, "base_estimator")
    cal = c.calibrators[0]
    folds.append({
        "coef": [round(float(v), R) for v in est.coef_[0]],
        "intercept": round(float(est.intercept_[0]), R),
        "a": float(cal.a_),
        "b": float(cal.b_),
    })

out = {
    "terms": list(tfidf.get_feature_names_out()),          # index order
    "idf": [round(float(v), R) for v in tfidf.idf_],
    "stop_words": sorted(tfidf.get_stop_words()),
    "sublinear_tf": bool(tfidf.sublinear_tf),
    "folds": folds,
}
with open("api/model.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))
print("terms:", len(out["terms"]), "folds:", len(folds))

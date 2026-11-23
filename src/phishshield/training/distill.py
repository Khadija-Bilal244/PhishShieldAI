"""DVC stage `distill` (proposal 5.4): distil the hybrid DistilBERT teacher into a CharCNN student and."""

from __future__ import annotations

import shutil
import time

import numpy as np
import torch

from phishshield.config import MODELS_DIR, REPORTS_DIR, ROOT, ensure_dirs, load_params, set_seed
from phishshield.evaluation.metrics import classification_metrics
from phishshield.features import char_tokenizer
from phishshield.features.lexical import FEATURE_NAMES
from phishshield.models.hybrid import load_hybrid, predict_logits
from phishshield.models.student import build_student, save_student, student_logits
from phishshield.training.calibration import fit_temperature_bounded, pick_threshold
from phishshield.training.common import load_split, sigmoid
from phishshield.training.student_trainer import fit_student
from phishshield.utils import tracking
from phishshield.utils.common import get_logger, save_json

log = get_logger("distill")
STUDENT_DIR = MODELS_DIR / "student"


def export_onnx(model, out_path, max_len: int, n_features: int) -> None:
    model.eval()
    ids = torch.zeros(2, max_len, dtype=torch.long)
    feats = torch.zeros(2, n_features, dtype=torch.float32)
    kwargs = dict(input_names=["char_ids", "features"], output_names=["logit", "embedding"],
                  dynamic_axes={"char_ids": {0: "batch"}, "features": {0: "batch"},
                                "logit": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, (ids, feats), str(out_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, (ids, feats), str(out_path), **kwargs)


def onnx_latency_ms(onnx_path, urls, Xs, max_len: int, n: int = 50) -> tuple[float, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ids = char_tokenizer.encode_batch(urls, max_len)
    full = sess.run(["logit"], {"char_ids": ids, "features": Xs.astype(np.float32)})[0]
    one = {"char_ids": ids[:1], "features": Xs[:1].astype(np.float32)}
    sess.run(["logit"], one)
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(["logit"], one)
    return (time.perf_counter() - t0) / n * 1000, full


def main() -> None:
    params = load_params()
    ensure_dirs()
    set_seed(params["seed"])
    scfg = params["student"]
    train, val, test = load_split("train"), load_split("val"), load_split("test")

    teacher, tmeta = load_hybrid(MODELS_DIR / "hybrid")
    log.info("Teacher soft labels on %d training URLs ...", len(train))
    t_logits = predict_logits(teacher, train.urls, train.Xs) / tmeta["temperature"]

    with tracking.run("distill-student", params["mlflow"]["experiment"],
                      {"model": "charcnn_student", **scfg}, {"stage": "distill"}):
        student = build_student(scfg, len(FEATURE_NAMES))
        info = fit_student(student, train, scfg, teacher_logits=t_logits, seed=params["seed"])

        v_logits = student_logits(student, val.urls, val.Xs, scfg["max_len"])
        temperature = fit_temperature_bounded(v_logits, val.y)
        threshold = pick_threshold(val.y, sigmoid(v_logits / temperature), params["evaluation"]["max_fnr"])

        n_params = sum(p.numel() for p in student.parameters())
        meta = {**{k: scfg[k] for k in ("max_len", "embed_dim", "num_filters", "kernel_sizes")},
                "n_features": len(FEATURE_NAMES), "temperature": temperature, "threshold": threshold,
                "n_params": n_params, "teacher_params": tmeta["n_params"]}
        if STUDENT_DIR.exists():
            shutil.rmtree(STUDENT_DIR)
        save_student(student, STUDENT_DIR, meta)
        export_onnx(student, STUDENT_DIR / "student.onnx", scfg["max_len"], len(FEATURE_NAMES))
        shutil.copy2(MODELS_DIR / "scaler.joblib", STUDENT_DIR / "scaler.joblib")
        rng = np.random.default_rng(params["seed"])
        bg = train.Xs[rng.choice(len(train), min(params["explain"]["shap_background"], len(train)), replace=False)]
        np.save(STUDENT_DIR / "shap_background.npy", bg.astype(np.float32))

        student_ms, onnx_logits = onnx_latency_ms(STUDENT_DIR / "student.onnx", test.urls, test.Xs, scfg["max_len"])
        torch_logits = student_logits(student, test.urls, test.Xs, scfg["max_len"])
        onnx_err = float(np.max(np.abs(onnx_logits - torch_logits)))
        s_prob = sigmoid(onnx_logits / temperature)
        test_m = classification_metrics(test.y, s_prob, threshold)

        t_prob = sigmoid(predict_logits(teacher, test.urls, test.Xs) / tmeta["temperature"])
        teacher_m = classification_metrics(test.y, t_prob, tmeta["threshold"])
        agreement = float(np.mean((t_prob >= tmeta["threshold"]) == (s_prob >= threshold)))
        predict_logits(teacher, test.urls[:1], test.Xs[:1])
        t0 = time.perf_counter()
        for _ in range(10):
            predict_logits(teacher, test.urls[:1], test.Xs[:1])
        teacher_ms = (time.perf_counter() - t0) / 10 * 1000

        report = {"student_test": test_m, "teacher_test": teacher_m, "teacher_student_agreement": agreement,
                  "onnx_max_abs_error": onnx_err, "student_params": n_params, "teacher_params": tmeta["n_params"],
                  "compression_ratio": tmeta["n_params"] / n_params,
                  "latency_ms": {"teacher_torch": teacher_ms, "student_onnx_cpu": student_ms},
                  "speedup": teacher_ms / max(student_ms, 1e-9),
                  "onnx_size_mb": (STUDENT_DIR / "student.onnx").stat().st_size / 1e6,
                  "temperature": temperature, "threshold": threshold, **info}
        tracking.log_metrics(test_m, prefix="test_")
        tracking.log_metrics({"agreement": agreement, "student_latency_ms": student_ms, "teacher_latency_ms": teacher_ms,
                              "compression_ratio": report["compression_ratio"], "speedup": report["speedup"],
                              "onnx_max_abs_error": onnx_err, "n_params": n_params})
        if tracking.tracking_enabled():
            import mlflow.pyfunc
            import pandas as pd

            from phishshield.models.student_pyfunc import StudentPyfunc

            mlflow.pyfunc.log_model(
                artifact_path="model", python_model=StudentPyfunc(),
                artifacts={"model_dir": str(STUDENT_DIR)},
                code_path=[str(ROOT / "src" / "phishshield")],
                input_example=pd.DataFrame({"url": test.urls[:3]}),
                pip_requirements=["onnxruntime", "scikit-learn", "pandas", "numpy", "joblib"])
        run_id = tracking.active_run_id()

    save_json(report, REPORTS_DIR / "student_metrics.json")
    save_json({"student": run_id}, REPORTS_DIR / "runs" / "student.json")
    log.info("Student F1=%.4f FNR=%.4f | teacher F1=%.4f | agreement=%.4f | %.1fx smaller, %.0fx faster (%.2f ms vs %.1f ms)",
             test_m["f1"], test_m["fnr"], teacher_m["f1"], agreement, report["compression_ratio"], report["speedup"],
             student_ms, teacher_ms)


if __name__ == "__main__":
    main()

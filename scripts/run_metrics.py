import fiftyone as fo
from fiftyone import ViewField as F

def run_metrics(view: fo.DatasetView, *, eval_prefix: str) -> None:
    # Optional: keep only samples that have both GT & pred for each task
    cls_view = view.exists("gt_type_cls").exists("pred_type_cls")
    reg_view = view.exists("gt_time_reg").exists("pred_time_reg")

    # --- Classification: accuracy + confusion matrix ---
    cls_results = cls_view.evaluate_classifications(
        "pred_type_cls",
        gt_field="gt_type_cls",
        eval_key=f"{eval_prefix}_type",
        missing="(missing)",   # if any labels are None :contentReference[oaicite:2]{index=2}
    )
    cls_results.print_metrics()
    cls_results.print_report()
    cls_results.plot_confusion_matrix()  # interactive plotly matrix :contentReference[oaicite:3]{index=3}

    # --- Regression: MAE/RMSE/etc ---
    reg_results = reg_view.evaluate_regressions(
        "pred_time_reg",
        gt_field="gt_time_reg",
        eval_key=f"{eval_prefix}_time",
    )
    reg_results.print_metrics()  # includes MAE/RMSE/etc :contentReference[oaicite:4]{index=4}


def quick_slice_stats(view: fo.DatasetView) -> dict[str, float]:
    # accuracy from your boolean field
    acc = view.mean("type_match")

    # MAE in seconds from your abs error scalar
    mae_sec = view.mean("abs_time_error_sec")

    # MAE in frames (you computed abs_time_error_frames)
    mae_frames = view.mean("abs_time_error_frames")

    return {"acc": acc, "mae_sec": mae_sec, "mae_frames": mae_frames}


dataset = fo.load_dataset("dada_eval_val")
view = dataset.match((F("json_valid") == True) & (F("abs_time_error_sec") <= 100.0))

run_metrics(view, eval_prefix="dada_eval_val")
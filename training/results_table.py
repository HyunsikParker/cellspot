"""Print the README results table from the metrics files, so the numbers in the text are the measured ones.

usage: python results_table.py CLASSIFIER_METRICS DETECTOR_METRICS
"""
import json
import sys


def pct(x):
    return f"{round(100 * x)}%"


def main():
    cls, det = (json.load(open(p)) for p in sys.argv[1:3])
    t, rule = cls["test"], cls.get("app_rule")
    vq, hq = det["VQ_test"], det["HQ_test"]
    f = vq["battery_images"]
    rows = [
        ("X-ray: is a battery inside?", f"{vq['images']} X-rays of {det.get('test_devices', '?')} devices",
         f"{f['found']} of {f['with_battery']} with a battery flagged ({pct(f['recall'])}), {f['missed']} missed, "
         f"{f['false_alarms']} flagged without one (score threshold {det['threshold']})"),
        ("X-ray: battery boxes", "same", f"AP50 {vq['battery_ap50']:.2f}; {hq['battery_ap50']:.2f} on the {hq['images']} high-quality X-rays"),
        ("Photo: naming the device", f"{t['n']} photos of {t['n']} devices",
         f"{pct(t['accuracy'])} named correctly, {pct(t['top2'])} right in the top two"),
    ]
    if rule:
        rows.append(("Photo: the app's rule", "same",
                     f"named {rule['named']} ({rule['named_wrong']} with a wrong name, still the right advice), "
                     f"asked the question on {rule['asked']}"))
    print("| Check | Test set | Result |\n|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()

from collections import defaultdict

METRICS_FOR_STEP = defaultdict(list)

def add_hacky_global_metrics(dict_to_add):
    global METRICS_FOR_STEP

    for k,v in dict_to_add.items():
        METRICS_FOR_STEP[f"{k}"].append(v)

def get_and_clear_hacky_global_metrics():
    global METRICS_FOR_STEP
    metrics = METRICS_FOR_STEP
    METRICS_FOR_STEP = defaultdict(list)
    return metrics

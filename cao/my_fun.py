import numpy as np
import pandas as pd

def parse_interaction(line):
    line_split = line.split(",")
    # keep just numeric and logical values
    symbolic_indexes = [1,2,3,41]
    clean_line_split = [item for i,item in enumerate(line_split) if i not in symbolic_indexes]
    return np.array([float(x) for x in clean_line_split])

def parse_interaction_with_key(line):
    import numpy as np

    line_split = line.split(",")
    # keep just numeric and logical values
    symbolic_indexes = [1,2,3,41]
    clean_line_split = [item for i,item in enumerate(line_split) if i not in symbolic_indexes]
    return (line_split[41], np.array([float(x) for x in clean_line_split]))

def summary_by_label(raw_data, label):
    from pyspark.mllib.stat import Statistics

    label_vector_data = raw_data.map(parse_interaction_with_key).filter(lambda x: x[0]==label)
    return Statistics.colStats(label_vector_data.values())


def get_variable_stats_df(stats_by_label, column_i):
    column_stats_by_label = [
        (stat[0], np.array([float(stat[1].mean()[column_i]), float(sqrt(stat[1].variance()[column_i])), float(stat[1].min()[column_i]), float(stat[1].max()[column_i]), int(stat[1].count())]))
        for stat in stats_by_label
        ]
    return pd.DataFrame.from_items(column_stats_by_label, columns=["Mean", "Std Dev", "Min", "Max", "Count"], orient='index')
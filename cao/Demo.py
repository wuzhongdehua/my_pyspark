from pyspark import SparkConf,SparkContext,SparkFiles

from pyspark.mllib.stat import Statistics
import pandas as pd
from math import sqrt
import numpy as np
from splearn.rdd import ArrayRDD
import os
import urllib
from time import time
import logging

class Demo:
    def __init__(self, master, name):
        self.name=name
        self.master=master

        print "init spark ..."
        os.environ["HADOOP_HOME"]="D:\code\wqr\hadoop-common-2.2.0-bin"
        conf = SparkConf()
        conf.setMaster(self.master)
        conf.setAppName(self.name)

        self.sc = SparkContext(conf=conf)

    def init_data(self):
        sc = self.sc
        print "load data ..."
        # urllib.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz", "kddcup.data_10_percent.gz")
        data_file = "./kddcup.data_10_percent.gz"
        self.raw_data = sc.textFile(data_file)

    def run1(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        raw_data = self.raw_data

        normal_raw_data = raw_data.filter(lambda x: 'normal.' in x)
        t0 = time()
        normal_count = normal_raw_data.count()
        tt = time() - t0
        logger.debug("There are {} 'normal' interactions".format(normal_count))
        logger.debug("Count completed in {} seconds".format(round(tt,3)))

        from pprint import pprint
        csv_data = raw_data.map(lambda x: x.split(","))
        t0 = time()
        head_rows = csv_data.take(5)
        tt = time() - t0
        logger.debug("Parse completed in {} seconds".format(round(tt,3)))
        pprint(head_rows[0])

    def run2(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        raw_data = self.raw_data
        raw_data_sample = raw_data.sample(False, 0.1, 1234)
        sample_size = raw_data_sample.count()
        total_size = raw_data.count()
        logger.debug("Sample size is {} of {}".format(sample_size, total_size))

        raw_data_sample_items = raw_data_sample.map(lambda x: x.split(","))
        sample_normal_tags = raw_data_sample_items.filter(lambda x: "normal." in x)

        # actions + time
        t0 = time()
        sample_normal_tags_count = sample_normal_tags.count()
        tt = time() - t0

        sample_normal_ratio = sample_normal_tags_count / float(sample_size)
        logger.debug("The ratio of 'normal' interactions is {}".format(round(sample_normal_ratio,3)))
        logger.debug("Count done in {} seconds".format(round(tt,3)))

    def run3(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        raw_data = self.raw_data
        # parse data
        csv_data = raw_data.map(lambda x: x.split(","))

        # separate into different RDDs
        normal_csv_data = csv_data.filter(lambda x: x[41]=="normal.")
        attack_csv_data = csv_data.filter(lambda x: x[41]!="normal.")

        normal_duration_data = normal_csv_data.map(lambda x: int(x[0]))
        attack_duration_data = attack_csv_data.map(lambda x: int(x[0]))

        total_normal_duration = normal_duration_data.reduce(lambda x, y: x + y)
        total_attack_duration = attack_duration_data.reduce(lambda x, y: x + y)

        logger.debug("Total duration for 'normal' interactions is {}". \
            format(total_normal_duration))
        logger.debug("Total duration for 'attack' interactions is {}". \
            format(total_attack_duration))

        normal_sum_count = normal_duration_data.aggregate(
            (0,0), # the initial value
            (lambda acc, value: (acc[0] + value, acc[1] + 1)), # combine value with acc
            (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])) # combine accumulators
        )

        print "Mean duration for 'normal' interactions is {}". \
            format(round(normal_sum_count[0]/float(normal_sum_count[1]),3))

        attack_sum_count = attack_duration_data.aggregate(
            (0,0), # the initial value
            (lambda acc, value: (acc[0] + value, acc[1] + 1)), # combine value with acc
            (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])) # combine accumulators
        )

        print "Mean duration for 'attack' interactions is {}". \
            format(round(attack_sum_count[0]/float(attack_sum_count[1]),3))

    def run4(self):
        from my_fun import parse_interaction,parse_interaction_with_key,summary_by_label

        raw_data = self.raw_data
        vector_data = raw_data.map(parse_interaction)
        # Compute column summary statistics.
        summary = Statistics.colStats(vector_data)

        print "Duration Statistics:"
        print " Mean: {}".format(round(summary.mean()[0],3))
        print " St. deviation: {}".format(round(sqrt(summary.variance()[0]),3))
        print " Max value: {}".format(round(summary.max()[0],3))
        print " Min value: {}".format(round(summary.min()[0],3))
        print " Total value count: {}".format(summary.count())
        print " Number of non-zero values: {}".format(summary.numNonzeros()[0])

        label_vector_data = raw_data.map(parse_interaction_with_key)
        normal_label_data = label_vector_data.filter(lambda x: x[0]=="normal.")

        normal_summary = Statistics.colStats(normal_label_data.values())

        print "Duration Statistics for label: {}".format("normal")
        print " Mean: {}".format(normal_summary.mean()[0],3)
        print " St. deviation: {}".format(round(sqrt(normal_summary.variance()[0]),3))
        print " Max value: {}".format(round(normal_summary.max()[0],3))
        print " Min value: {}".format(round(normal_summary.min()[0],3))
        print " Total value count: {}".format(normal_summary.count())
        print " Number of non-zero values: {}".format(normal_summary.numNonzeros()[0])

        normal_sum = summary_by_label(raw_data, "normal.")

        print "Duration Statistics for label: {}".format("normal")
        print " Mean: {}".format(normal_sum.mean()[0],3)
        print " St. deviation: {}".format(round(sqrt(normal_sum.variance()[0]),3))
        print " Max value: {}".format(round(normal_sum.max()[0],3))
        print " Min value: {}".format(round(normal_sum.min()[0],3))
        print " Total value count: {}".format(normal_sum.count())
        print " Number of non-zero values: {}".format(normal_sum.numNonzeros()[0])

        label_list = ["back.","buffer_overflow.","ftp_write.","guess_passwd.",
                      "imap.","ipsweep.","land.","loadmodule.","multihop.",
                      "neptune.","nmap.","normal.","perl.","phf.","pod.","portsweep.",
                      "rootkit.","satan.","smurf.","spy.","teardrop.","warezclient.",
                      "warezmaster."]
        stats_by_label = [(label, summary_by_label(raw_data, label)) for label in label_list]

        duration_by_label = [
            (stat[0], np.array([float(stat[1].mean()[0]), float(sqrt(stat[1].variance()[0])), float(stat[1].min()[0]), float(stat[1].max()[0]), int(stat[1].count())]))
            for stat in stats_by_label]

        pd.set_option('display.max_columns', 50)

        stats_by_label_df = pd.DataFrame.from_items(duration_by_label, columns=["Mean", "Std Dev", "Min", "Max", "Count"], orient='index')

        print "Duration statistics, by label"
        stats_by_label_df
import json
import logging
import os
import subprocess
import sys
import time

from drain3 import TemplateMiner

logger = logging.getLogger(__name__)
input_dir = '../data/'
in_log_file = "sample_hdfs.log"
output_dir = 'drainResult/'


template_miner = TemplateMiner()

line_count = 0
start_time = time.time()
batch_start_time = start_time
batch_size = 10000
with open(input_dir + in_log_file) as f:
    for line in f:
        line = line.rstrip()
        line = line.partition(": ")[2]
        result = template_miner.add_log_message(line)
        line_count += 1
        if result["change_type"] != "none":
            result_json = json.dumps(result)
sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
logging.basicConfig(filename=output_dir + in_log_file + '_templates.csv', filemode='w', level=logging.INFO, format='%(message)s')
for cluster in sorted_clusters:
    logger.info(cluster)




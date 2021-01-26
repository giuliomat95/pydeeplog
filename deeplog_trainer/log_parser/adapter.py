import re


class BatrasioAdapter:
    def __init__(self, delimiter='TCP source connection created',
                 anomaly1='TCP source SSL error',
                 anomaly2='TCP source socket error'):
        self.d = {}
        self.last_session_id = 0
        self.anomaly_flag = {}
        self.delimiter = delimiter
        self.anomaly1 = anomaly1
        self.anomaly2 = anomaly2

    def get_session_id(self, log_msg: str):
        """
        Given the log message, returns the corresponding session Id it belongs
        to.
        """
        log_msg = log_msg.rstrip()
        procid = re.search(r"^(\d+)", log_msg)[0]
        if procid not in self.d.keys():
            self.d[procid] = self.last_session_id + 1
            self.anomaly_flag[self.d[procid]] = False
        if self.delimiter in log_msg:
            self.d[procid] = self.last_session_id + 1
            self.last_session_id += 1
            self.anomaly_flag[self.d[procid]] = False
        if self.anomaly1 in log_msg or self.anomaly2 in log_msg:
            self.anomaly_flag[self.d[procid]] = True
        return self.d[procid], self.anomaly_flag

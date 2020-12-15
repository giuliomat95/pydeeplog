import re

class BatrasioAdapter:
    def __init__(self):
        self.d = {}
        self.last_session_id = 0
        self.anomaly_flag = {}

    def get_sessionId(self, log_msg: str):
        log_msg = log_msg.rstrip()
        procid = re.search(r"^(\d+)", log_msg)[0]
        if procid not in self.d.keys():
            self.d[procid] = self.last_session_id + 1
            self.anomaly_flag[self.d[procid]] = False
        delimiter = 'TCP source connection created'
        anomaly1 = 'TCP source SSL error'
        anomaly2 = 'TCP source socket error'
        if delimiter in log_msg:
            self.d[procid] = self.last_session_id + 1
            self.last_session_id += 1
            self.anomaly_flag[self.d[procid]] = False
        if anomaly1 in log_msg or anomaly2 in log_msg:
            self.anomaly_flag[self.d[procid]] = True
        return self.d[procid], self.anomaly_flag

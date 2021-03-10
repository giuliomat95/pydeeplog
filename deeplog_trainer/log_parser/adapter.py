import re
from datetime import datetime
from datetime import timedelta


class SessionAdapter:
    def __init__(self, logformat, delimiter: str = None,
                 anomaly_labels: [] = None, regex=None, time_format: str = None,
                 delta: dict = None):
        """
        :param logformat: format of the entry log. The variable must contain the
        word 'Content' in order to isolate the part to be parsed by Drain.
        :param delimiter: If necessary, the string sentence that calls a new
        session.
        :param anomaly_labels: list with the strings leading to an anomaly
        :param regex: The session identifier in regex format. For example, it
        could be the id of a particular process.
        :param time_format: If the sessions needs to be grouped calculating the
        time elapsed between the entry log and the previous one, set the format
        of time. Example: HH:MM:SS.
        :param delta: dictionary indicating the time that must elapse to create
        a new session. Example: {'minutes'=1, 'seconds'=30}
        """
        self.d = {}
        self.last_session_id = 0
        self.anomaly_flag = {}
        self.delimiter = delimiter
        self.anomaly_labels = anomaly_labels
        self.regex = regex
        self.logformat = logformat
        self.delta = delta
        self.time_format = time_format
        # Content word must be in logformat to isolate the log part to be parsed
        # by Drain
        assert 'Content' in logformat

    def get_session_id(self, log: str):
        """
        Given the log message, returns the corresponding session Id it belongs
        to.
        """
        log = log.rstrip()
        if self.regex:
            identifier = re.search(self.regex, log)[0]
            if identifier not in self.d.keys():
                self.d[identifier] = self.last_session_id + 1
                self.anomaly_flag[self.d[identifier]] = False
                if not self.delimiter:
                    self.last_session_id += 1
            if self.delimiter and self.delimiter in log:
                self.d[identifier] = self.last_session_id + 1
                self.last_session_id += 1
                self.anomaly_flag[self.d[identifier]] = False
            if not self.anomaly_flag[self.d[identifier]]:
                self.anomaly_flag[self.d[identifier]] = any(anomaly_label in log
                                                            for anomaly_label in
                                                            self.anomaly_labels)
            return self.d[identifier], self.anomaly_flag
        elif self.delta:
            assert 'Time' in self.logformat
            starter_time = 0
            delta = timedelta(**self.delta)
            headers, regex = self.generate_logformat_regex()
            match = regex.search(log.strip())
            time = match.group('Time')
            if not starter_time:
                starter_time = time
            elapsed_time = \
                datetime.strptime(time, self.time_format)-datetime.strptime(
                    starter_time, self.time_format)
            if elapsed_time > delta:
                self.last_session_id += 1
                self.anomaly_flag[self.last_session_id] = False
            if not self.anomaly_flag[self.last_session_id]:
                self.anomaly_flag[self.last_session_id] = \
                    any(anomaly_label in log for anomaly_label in
                        self.anomaly_labels)
            return self.last_session_id
        elif not self.regex:
            # If no identifier set him to default value of 0
            if self.delimiter in log:
                self.last_session_id += 1
                self.anomaly_flag[self.last_session_id] = False
            if not self.anomaly_flag[self.last_session_id]:
                self.anomaly_flag[self.last_session_id] = \
                    any(anomaly_label in log for anomaly_label in
                        self.anomaly_labels)
            return self.last_session_id
        else:
            raise Exception('Error')

    def generate_logformat_regex(self):
        """
        Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', self.logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

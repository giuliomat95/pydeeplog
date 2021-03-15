import re
from datetime import datetime
from datetime import timedelta
from abc import ABCMeta, abstractmethod


class SessionAdapterInterface(metaclass=ABCMeta):
    """Abstract Sessions Factory Interface"""

    @staticmethod
    @abstractmethod
    def get_session_id(log: str):
        """
        The static Abstract factory interface method: given the log message,
        returns the corresponding session Id it belongs to
        """
        pass


class OnlyIdentifier(SessionAdapterInterface):
    """A Concrete Class that implements the SessionAdapter interface"""

    def __init__(self, regex, anomaly_labels: [] = None):
        """
        :param regex: The session identifier in format “r-string”. For example,
        it could be the id of a particular process.
        :param anomaly_labels: List with the strings leading to an anomaly
        """
        self.regex = regex
        self.anomaly_labels = anomaly_labels
        self.d = {}
        self.last_session_id = 0
        self.anomaly_flag = {}
        # 'Content' word must be in logformat variable to isolate the log part
        # to be parsed by Drain

    def get_session_id(self, log: str):
        log = log.rstrip()
        identifier = re.search(self.regex, log)[0]
        if identifier not in self.d.keys():
            self.d[identifier] = self.last_session_id + 1
            self.last_session_id += 1
            if self.anomaly_labels:
                self.anomaly_flag[self.d[identifier]] = False
        if self.anomaly_labels and not self.anomaly_flag[self.d[identifier]]:
            self.anomaly_flag[self.d[identifier]] = any(anomaly_label in log
                                                        for anomaly_label in
                                                        self.anomaly_labels)
        return self.d[identifier], self.anomaly_flag


class IdentifierAndDelimiter(SessionAdapterInterface):
    """A Concrete Class that implements the SessionAdapter interface"""

    def __init__(self, regex, delimiter: str, anomaly_labels: [] = None):
        """
        :param regex: The identifier in format “r-string”. For example,
        it could be the id of a particular process.
        :param delimiter: The string sentence that calls a new session.
        :param anomaly_labels:  List with the strings leading to an anomaly
        """
        self.regex = regex
        self.delimiter = delimiter
        self.d = {}
        self.last_session_id = 0
        self.anomaly_flag = {}

        self.anomaly_labels = anomaly_labels


    def get_session_id(self, log: str):
        log = log.rstrip()
        identifier = re.search(self.regex, log)[0]
        if identifier not in self.d.keys():
            self.d[identifier] = self.last_session_id + 1
            if self.anomaly_labels:
                self.anomaly_flag[self.d[identifier]] = False
        if self.delimiter in log:
            self.d[identifier] = self.last_session_id + 1
            self.last_session_id += 1
            if self.anomaly_labels:
                self.anomaly_flag[self.d[identifier]] = False
        if self.anomaly_labels and not self.anomaly_flag[self.d[identifier]]:
            self.anomaly_flag[self.d[identifier]] = any(anomaly_label in log
                                                        for anomaly_label in
                                                        self.anomaly_labels)
        return self.d[identifier], self.anomaly_flag


class OnlyDelimiter(SessionAdapterInterface):
    """A Concrete Class that implements the SessionAdapter interface"""

    def __init__(self, delimiter: str, anomaly_labels: [] = None):
        """
        :param delimiter: The string sentence that calls a new session.
        :param anomaly_labels: List with the strings leading to an anomaly
        """
        self.last_session_id = 0
        self.anomaly_flag = {}
        self.delimiter = delimiter
        self.anomaly_labels = anomaly_labels

    def get_session_id(self, log: str):
        log = log.rstrip()
        if self.delimiter in log:
            self.last_session_id += 1
            if self.anomaly_labels:
                self.anomaly_flag[self.last_session_id] = False
        if self.anomaly_labels and not self.anomaly_flag[self.last_session_id]:
            self.anomaly_flag[self.last_session_id] = any(anomaly_label in log
                                                          for anomaly_label in
                                                          self.anomaly_labels)
        return self.last_session_id, self.anomaly_flag


class TimeInterval(SessionAdapterInterface):
    """A Concrete Class that implements the SessionAdapter interface"""

    def __init__(self, logformat, time_format: str, delta: dict,
                 anomaly_labels: [] = None):
        """
        :param logformat: Format of the entry log. The variable must contain the
        word 'Time' to calculate the time interval between the entry log and
        another one.
        :param time_format: Set the format of time. Example: '%H:%M:%S.%f'.
        :param delta: Dictionary indicating the time that must elapse to create
        a new session. Example: {'minutes'=1, 'seconds'=30}
        :param anomaly_labels: List with the strings leading to an anomaly.
        """
        self.logformat = logformat
        assert 'Time' in self.logformat
        self.last_session_id = 0
        self.anomaly_flag = {}
        self.delta = delta
        self.time_format = time_format
        self.starter_time = 0
        self.anomaly_labels = anomaly_labels

    def get_session_id(self, log: str):
        delta = timedelta(**self.delta)
        headers, regex = ParseMethods.generate_logformat_regex(self.logformat)
        match = regex.search(log.strip())
        time = match.group('Time')
        if not self.starter_time:
            self.starter_time = time
            self.last_session_id = 1
        elapsed_time = \
            datetime.strptime(time, self.time_format) - datetime.strptime(
                self.starter_time, self.time_format)
        if elapsed_time >= delta:
            self.last_session_id += 1
            self.starter_time = time
            if self.anomaly_labels:
                self.anomaly_flag[self.last_session_id] = False
        if self.anomaly_labels and not self.anomaly_flag[
            self.last_session_id
        ]:
            self.anomaly_flag[self.last_session_id] = \
                any(anomaly_label in log for anomaly_label in
                    self.anomaly_labels)
        return self.last_session_id, self.anomaly_flag


class ParseMethods:
    def __init__(self):
        pass

    @staticmethod
    def generate_logformat_regex(logformat):
        """
        Function to generate regular expression to split log messages
        """
        # 'Content' word must be in logformat variable to isolate the log part
        # to be parsed by Drain
        assert 'Content' in logformat
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
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


class AdapterFactory:
    """The factory class"""
    ADAPTER_TYPE_DELIMITER = 'only_delimiter'
    ADAPTER_TYPE_DELIMITER_AND_REGEX = 'delimiter+regex'
    ADAPTER_TYPE_REGEX = 'only_identifier'
    ADAPTER_TYPE_INTERVAL_TIME = 'interval_time'

    def instantiate_product(self, adapter_type: str, **kwargs):
        try:
            if adapter_type == AdapterFactory.ADAPTER_TYPE_DELIMITER:
                self._validate_delimiter_kwargs(kwargs)
                return OnlyDelimiter(**kwargs)
            if adapter_type == AdapterFactory.ADAPTER_TYPE_REGEX:
                self._validate_regex_kwargs(kwargs)
                return OnlyIdentifier(**kwargs)
            if adapter_type == AdapterFactory.ADAPTER_TYPE_DELIMITER_AND_REGEX:
                self._validate_regex_and_delimiter_kwargs(kwargs)
                return IdentifierAndDelimiter(**kwargs)
            if adapter_type == AdapterFactory.ADAPTER_TYPE_INTERVAL_TIME:
                self._validate_time_interval_kwargs(kwargs)
                return TimeInterval(**kwargs)
            raise Exception('Adapter type not found')
        except Exception as _e:
            print(_e)
        return None

    def _validate_delimiter_kwargs(self, kwargs):
        if not 'delimiter' in kwargs:
            raise ValueError('Provide right parameters')

    def _validate_regex_kwargs(self, kwargs):
        if not 'regex' in kwargs:
            raise ValueError('Provide right parameters')

    def _validate_regex_and_delimiter_kwargs(self, kwargs):
        if not {'delimiter', 'regex'}.issubset(set(kwargs)):
            raise ValueError('Provide right parameters')

    def _validate_time_interval_kwargs(self, kwargs):
        if not {'logformat', 'time_format', 'delta'}.issubset(set(kwargs)):
            raise ValueError('Provide right parameters')

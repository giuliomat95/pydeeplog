from deeplog_trainer.log_parser.adapter import SessionAdapter
import pytest
import pdb


@pytest.fixture(scope='session')
def batrasio_adapter():
    return SessionAdapter(logformat='<Pid>  <Content>',
                          delimiter='TCP source connection created',
                          anomaly_labels=['TCP source SSL error',
                                          'TCP source socket error'],
                          regex=r"^(\d+)")


def get_batrasio_data():
    expected_sess_id = [1, 1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3]
    expected_anomaly_flag = [False, False, False, False, False, True, True,
                             False, True, False, False, False, False]
    with open('data/sample_test_batrasio.log') as f:
        for i, line in enumerate(f):
            yield line.strip(), expected_sess_id[i], expected_anomaly_flag[i]


@pytest.mark.parametrize("logs, expected_sess_id, expected_anomaly_flag",
                         get_batrasio_data())
def test_batrasio_sessions(logs, expected_sess_id, expected_anomaly_flag,
                           batrasio_adapter):
    sess_id, anomaly_flag = batrasio_adapter.get_session_id(log=logs)
    # Verify the result correspond to what we expect
    assert sess_id == expected_sess_id
    # The flag indicator of the anomalous nature of the sessions must be a
    # boolean
    assert anomaly_flag[sess_id] == expected_anomaly_flag


@pytest.fixture(scope='session')
def hdfs_adapter():
    return SessionAdapter(logformat='<Date> <Time> <Pid> <Level> '
                                    '<Component>: <Content>',
                          regex=r'blk_-?\d+')


def get_hdfs_data():
    expected_sess_id = [1, 1, 2, 2, 2, 1, 1, 2, 3, 4, 1, 4, 3, 3, 3]
    expected_block_ids = {'blk_-1608999687919862906',
                          'blk_7503483334202473044',
                          'blk_-3544583377289625738',
                          'blk_-9073992586687739851'}
    with open('data/sample_test_hdfs.log') as f:
        for i, line in enumerate(f):
            yield line.strip(), expected_sess_id[i], expected_block_ids


@pytest.mark.parametrize("logs, expected_sess_id, expected_block_ids",
                         get_hdfs_data())
def test_hdfs_sessions(logs, expected_sess_id, expected_block_ids,
                       hdfs_adapter):
    sess_id, anomaly_flag = hdfs_adapter.get_session_id(log=logs)
    # Verify the result correspond to what we expect
    assert sess_id == expected_sess_id
    assert set(hdfs_adapter.d.keys()).issubset(expected_block_ids)


@pytest.fixture(scope='session')
def no_procid_adapter():
    return SessionAdapter(logformat='<Content>',
                          delimiter='TCP source connection created',
                          anomaly_labels=['TCP source SSL error',
                                          'TCP source socket error'])


def get_no_procid_data():
    expected_sess_id = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    expected_anomaly_flag = [False, False, False, False, False, False, False,
                             False, False, False, False, False, False]
    with open('data/sample_test_no_procid.log') as f:
        for i, line in enumerate(f):
            yield line.strip(), expected_sess_id[i], expected_anomaly_flag[i]


@pytest.mark.parametrize("logs, expected_sess_id, expected_anomaly_flag",
                         get_no_procid_data())
def test_no_procid_sessions(logs, expected_sess_id, expected_anomaly_flag,
                            no_procid_adapter):
    sess_id, anomaly_flag = no_procid_adapter.get_session_id(log=logs)
    # Verify the result correspond to what we expect
    assert sess_id == expected_sess_id
    assert anomaly_flag[sess_id] == expected_anomaly_flag


@pytest.fixture(scope='session')
def box_unix_adapter():
    return SessionAdapter(logformat='<Date> <Time>, <Content>',
                          time_format='%H:%M:%S.%f', delta={'milliseconds': 2})


def get_box_unix_data():
    expected_sess_id = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6]
    with open('data/sample_test_box_unix.log') as f:
        for i, line in enumerate(f):
            yield line.strip(), expected_sess_id[i]


@pytest.mark.parametrize("logs, expected_sess_id", get_box_unix_data())
def test_box_unix_sessions(logs, expected_sess_id, box_unix_adapter):
    sess_id, anomaly_flag = box_unix_adapter.get_session_id(log=logs)
    # Verify the result correspond to what we expect
    assert sess_id == expected_sess_id

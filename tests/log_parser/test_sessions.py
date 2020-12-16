import unittest
import sys

sys.path.append('../../')
from unittest_data_provider import data_provider
from drain3 import TemplateMiner
from deeplog_trainer.log_parser.adapter import BatrasioAdapter
from deeplog_trainer.log_parser.sessions import SessionStorage
from deeplog_trainer.log_parser.drain import Drain
import pytest
adapter = BatrasioAdapter()
@pytest.mark.parametrize("logs,expected", [
        ('2337  server=0.0.0.0: connector=null, client=34.246.191.216:44940: TCP source connection created', 1),
        ('2337	server=0.0.0.0: TLS source connection from 34.246.191.216:44940', 1),
        ('2337	Multi target connection created', 1),
        ('2386	server=0.0.0.0: connector=null, client=213.192.202.200:36992: TCP source connection created', 2),
        ('2386	server=0.0.0.0: TLS source connection from 213.192.202.200:36992', 2),
        ('2386	server=0.0.0.0: event_type=AUTH, connector=null, client=213.192.202.200:36992: TCP source SSL error: Error: 139995610654592:error:140890C7:SSL routines:ssl3_get_client_certificate:peer did not return a certificate:../deps/openssl/openssl/ssl/s3_srvr.c:3346:', 2),
        ('2386	server=0.0.0.0: Plain source connection from 52.19.12.121:47372', 2),
        ('2379  server=0.0.0.0: connector=null, client=52.19.12.121:49422: TCP source connection created', 3),
        ('2386  connector=52541011, kind=proto-piped-N[ntf], path=51.141.33.185:45068 < ->10.11.24.33:443 <= > null < ->null, protos=ntf: Connector closed', 2),
        ('2379  connector=52483975, kind=unpiped, path=52.19.12.121:49422 < ->10.11.24.33:1515 <= > 10.11.24.33:39229 < ->172.17.12.211:1515:  Connector created', 3),
        ('2379  connector=52483972, kind=unpiped, path=52.86.47.45:45320 < ->10.11.24.33:443 <= > 10.11.24.33:57724 < ->172.17.12.110:660: Connector closed', 3),
        ('2379  connector=52483972, kind=unpiped, path=52.86.47.45:45320 < ->10.11.24.33:443 <= > 10.11.24.33:57724 < ->172.17.12.110:660: Connector closed', 3),
        ('2379  target= / etc / logtrust / batrasio / run / targets / 172.17.12.55: Trying connection to port 1515', 3)])

def test_get_sessionId(logs, expected):
    assert adapter.get_sessionId(log_msg=logs)[0] == expected

"""
class SessionStorageTest(unittest.TestCase):
    def setUp(self):
        self.adapter = BatrasioAdapter()
        template_miner = TemplateMiner()
        self.drain = Drain(template_miner)
        self.storage = SessionStorage()
"""

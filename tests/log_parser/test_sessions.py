import unittest
from unittest_data_provider import data_provider

from deeplog_trainer.log_parser import sessions

class DimasSessionStorageTest(unittest.TestCase):
    def setUp(self):
        # Nothing
        pass

    def msg_provider():
        return (
            (
                ['connector=52541013: Closing multi target connection',
                'connector=52540976, kind=unpiped, path=52.86.47.45:58170<->10.11.24.33:443<=>10.11.24.33:41904<->172.17.12.211:660: Connector closed',
                'connector=52541014, kind=unpiped, path=52.19.12.121:49440<->10.11.24.33:1515<=>10.11.24.33:54193<->172.17.12.65:1515: Connector closed'],
            ),
        )

    @data_provider(msg_provider)
    def test_add_get(self, msg_list):
        storage = sessions.DimasSessionStorage()

        for msg_line in msg_list:
            storage.add(msg_line)

        self.assertEqual(len(msg_list), len(storage.get_all()), 'incorrect size')


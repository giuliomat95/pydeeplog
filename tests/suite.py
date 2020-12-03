import os
import sys
import unittest

tests_path = os.path.dirname(os.path.realpath(__file__))

src_path = '..'
sys.path.insert(0, os.path.abspath(os.path.join(tests_path, src_path)))

def get_suite():
    loader = unittest.TestLoader()
    suite = loader.discover(tests_path)
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(get_suite())

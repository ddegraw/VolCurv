from __future__ import print_function

import os
import sys
from pprint import pprint

import requests
"""
ENV_VARS = ['QUANT_USER', 'QUANT_PASS']
if not set(ENV_VARS).issubset(set(os.environ)):
    print("Environment variables %s not set. Exiting..." % ENV_VARS)
    exit()
USER = os.environ['QUANT_USER']
PASS = os.environ['QUANT_PASS']
"""

USER = 'daiwa'
PASS = 'gQUK#ag@27_k@zWE'

HOST = 'https://quant.reactive.ai'


class RestClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.auth = (USER, PASS)

    def query(self, timestamp, symbol):
        resp = self.session.get(HOST + '/v1/query_model?timestamp={}&symbol={}'.format(timestamp, symbol))
        if resp.status_code == 401:
            print("wrong user and or password")
            exit()
        response = resp.json()
        pprint(response)


if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) == 3:
        ts = sys.argv[1]
        sym = sys.argv[2]
    else:
        ts = '2016-11-24%2011:20:00'
        sym = '7741'
    restClient = RestClient()
    print('ts =', ts, 'symbol =', sym)
    restClient.query(timestamp=ts, symbol=sym)

#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='')
parser.add_argument('-c', '--config', default='config.ini', help='config.ini file path')
parser.add_argument('-t', '--test', help='to test rdfize/rdfloader')

subparsers = parser.add_subparsers()
parser_rdfizer = subparsers.add_parser('rdfizer')
parser_rdfizer.add_argument('-i', '--in_path', nargs=1, required=True, help='input data file path')
parser_rdfizer.add_argument('-o', '--out_path', nargs=1, required=True, help='output data file path')

parser_rdfloader = subparsers.add_parser('rdfloader')
parser_rdfloader.add_argument('-i', '--in_path', nargs=1, required=True, help='input rdf file path')
parser_rdfloader.add_argument('-db', '--db_path', nargs=1, required=True, help='database path')

args = parser.parse_args()
print(args)

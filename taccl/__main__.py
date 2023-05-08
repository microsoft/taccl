#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from taccl.cli import *

import argparse
import argcomplete
import sys

def main():
    parser = argparse.ArgumentParser('taccl')

    cmd_parsers = parser.add_subparsers(title='command', dest='command')
    cmd_parsers.required = True

    handlers = []
    handlers.append(make_handle_solve_comm_sketch(cmd_parsers))
    handlers.append(make_handle_combine_comm_sketch(cmd_parsers))
    handlers.append(make_handle_ncclize(cmd_parsers))

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    for handler in handlers:
        if handler(args, args.command):
            break

if __name__ == '__main__':
    main()

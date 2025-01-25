#!/usr/bin/env python
import argparse
import sys

# torchlight
from torchlight.torchlight.io import import_class



def classify(arg, datas):  # 主函数，接收第一组的输入
    # parser = argparse.ArgumentParser(description='Processor collection')
    #
    # # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    # processors['demo_old'] = import_class('processor.demo_old.Demo')
    # processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    # processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    # #endregion yapf: enable
    #
    # # add sub-parser
    # subparsers = parser.add_subparsers(dest='processor')
    # for k, p in processors.items():
    #     subparsers.add_parser(k, parents=[p.get_parser()])
    #
    # # read arguments
    # arg = parser.parse_args() todo 把接收函数放在第一组那里

    # start
    Processor = processors['recognition']  # 有很多个不同的processer，根据需要选
    p = Processor(sys.argv[2:])

    p.start()

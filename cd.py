# -*- coding: utf-8 -*-
print('hello world')


# Ipython debug前先运行 cd.py
if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
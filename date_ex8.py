#!/usr/bin/env python
# coding=utf-8

import datetime
def dt_ex8():
    now_time = datetime.datetime.now()
    yes_time = now_time + datetime.timedelta(days=-8)
    return yes_time.strftime('%Y%m%d')
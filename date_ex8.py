# coding=utf-8

import pandas as pd
import datetime

now_time = datetime.datetime.now()
yes_time = now_time + datetime.timedelta(days=-7)
print yes_time.strftime('%Y-%m-%d')
pass
import time

import pandas as pd
im
from pytz import timezone, tzinfo

from datetime import datetime, timedelta, timezone

# unix time is number of seconds sice 01-01-1970
unix_stamp = 1545730073
print({time.time()})
print(f"{time.gmtime(unix_stamp)}")
print(f"{time.localtime(unix_stamp)}")
print(f"{time.daylight}")


print(f"{datetime.fromtimestamp(time.time())}")
print(f"{datetime.now()}")
print(f"{datetime.now(timezone.utc)}")
print(f"{datetime.now()}")



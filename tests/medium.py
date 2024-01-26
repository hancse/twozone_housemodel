# See: https://medium.com/swlh/making-sense-of-timezones-in-python-16d8ae210c1c

from datetime import datetime
from pytz import timezone

# set the time to noon on 19-08-2019 (naive)
naive = datetime(2019, 8, 19, 12, 00, 00)
print(f"Naive datetime: {naive}")

# Let's treat this time as being in the utc timezone
aware = timezone('utc').localize(naive)
print(f"Aware datetime: {aware}")

# Let's treat this time as being in the UTC timezone
aware = timezone('UTC').localize(naive)
print(f"Aware datetime: {aware}")

# Let's treat this time as being in the timezone Europe/Amsterdam
aware = timezone('Europe/Amsterdam').localize(naive)
print(f"Aware datetime: {aware}")

UTC = timezone('UTC')
NYC = timezone('America/New_York')

aware = UTC.localize(naive)

# UTC is 5 hours ahead of New York
local_time_in_New_York = aware.astimezone(NYC)
print(f"{local_time_in_New_York}")

# Avoid using datetime.replace(tzinfo=…) to make timezones aware
# It will only work with UTC and the behavior
# is inconsistent with other timezones. Rather use localize() as above.

# Getting the correct UTC time

from datetime import datetime
from pytz import UTC

# naive UTC
naive = datetime.utcnow()
print(f"Naive: {naive}")

aware = UTC.localize(datetime.utcnow())
# Using datetime.now() just returns your system’s current wall-clock time.
print(f"{aware}")



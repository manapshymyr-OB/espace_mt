import asf_search as asf
from dotenv import load_dotenv
import os
import pickle

from utils import get_data

# load env variables
load_dotenv()
token = os.getenv('eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Im1hbmFwX3NoeW15ciIsImV4cCI6MTcwNTgxNjg2MiwiaWF0IjoxNzAwNjMyODYyLCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.MGQzgTh78YvdSYmzueMTiPHzcbCYDJKgUFV2ees9KIrhQ6VbXl57BZjqbFaSkY3oTe894kvZEiEAVcUm1oL0AlOYh4nDLcDQfriAK9WLX7eUnN62bwFjN9wrQfXhsGpdHX2LXUcd0lZvlx17q2PsPaJXU8ejdH603z7WWCXZbs4_y6x0H2OEwW-Es8CZAtPSO5tp8t9O_869YHhdsiw9wcJ8R9ciG89wGHlzymElMOOq0UTIhfuqS1V4EvdOqo_64rgL6_If_YHMwgo2XktPq-eNrxjF8D2B1bap-WIpm_9DFWhZnneGHQ_s1ywAxq-il3KF6Y3KKiv5g8AKA_m5HA')
session = asf.ASFSession().auth_with_token(token)

# downloads data starting from start date until the end date
start_date = '2023-06-'
end_date = 'today'

# processing level
processingLevel = 'GRD_HD'
beamMode = 'IW'
flightDirection = 'DESCENDING'


def main():
    # search for data
    results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1],
                             intersectsWith=wkt,
                             # polarization = 'VV',
                             beamMode=beamMode,
                             processingLevel=processingLevel,
                             end=end_date,
                             start=start_date,
                             flightDirection=flightDirection)

    # save downloaded names into the pickle
    if os.path.exists(os.path.join(c.resources_dir, 'results.pkl')):
        dbfile = open(os.path.join(c.resources_dir, 'results.pkl'), 'rb')
        downloaded_products = pickle.load(dbfile)
        dbfile.close()
    else:
        downloaded_products = []

    total = 0
    urls = []
    names = []

    for r in results:
        total += r.properties['bytes']

        # print(r.properties['fileName'])

        if r.properties['fileName'] not in downloaded_products:
            urls.append(r.properties['url'])
        names.append(r.properties['fileName'])

    asf.download_urls(urls=urls, path=download_dir, session=session, processes=5)

    # save downloaded names into the pickle
    dbfile = open(os.path.join(c.resources_dir, 'results.pkl'), 'ab')

    # source, destination
    pickle.dump(names, dbfile)
    dbfile.close()
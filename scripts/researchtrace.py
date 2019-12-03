import urllib2
import re
from bs4 import BeautifulSoup as bs
import traceback
import requests
import unidecode
import time
import random

_HOST = 'https://scholar.google.com'
_AUTHSEARCH = '/citations?view_op=search_authors&hl=en&mauthors={0}'
_CITATIONAUTH = '/citations?user={0}&hl=en'

keyword='refactor'
data= urllib2.urlopen("https://esec-fse19.ut.ee/committees/research-papers/").read()


def fetch_author(author):
    url = _AUTHSEARCH.format(requests.utils.quote(unidecode.unidecode(author)))
    page = bs(urllib2.urlopen(_HOST+url).read(), 'html.parser')
    id=re.findall("user=[a-zA-Z0-9\-_]*", page.prettify())
    # if id is not None and len(id) > 0:
    #     print(id[0].split("=")[1])
    return id[0].split("=")[1]


def search_pubs(author, author_id, query):
    url = _CITATIONAUTH.format(author_id)
    page = bs(urllib2.urlopen(_HOST + url).read(), "html.parser")
    pubs = page.select('#gsc_a_b td a')
    for p in pubs:
        if p.has_attr('data-href'):
            #print(p['data-href'])
            title, abstract = parse_pub(p['data-href'])
            if query in title or query in abstract:
                print("Author: " + author)
                print("Title: " + title)


def parse_pub(pub_url):
    page = bs(urllib2.urlopen(_HOST+pub_url), "html.parser")
    #print(page.prettify())
    title=page.select('#gsc_vcd_title > a')[0].text
    # print(title)
    abstract=page.select('#gsc_vcd_descr')[0].text
    # print(abstract)
    return title, abstract


soup = bs(data, 'html.parser')
for x in soup.select('#post-293 > div > section.mpl-section-service.mpl-elm.mpl-css-136976.mpl-section.mpl-section-inner > div > div > div.mpl-section-content-wrap > div > ul   a'):
    researcher=x.text.split(',')[0]
    print(researcher)
    if researcher is not None:
        try:
            id = fetch_author(researcher)
            search_pubs(researcher, id, keyword)
            time.sleep(5 + random.uniform(0, 5))
        except:
            traceback.print_exc()
            continue
        #     search_query = scholarly.search_author(str(researcher))
        #     author=next(search_query).fill()
        #     for pub in author.publications[:10]:
        #         if keyword in pub.bib['abstract']:
        #             print(pub.bib['title'])
        # except Exception as e:
        #     traceback.print_exc()

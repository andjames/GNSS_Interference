"""
Search the CDDIS directory listing for specific stations
to find their exact filename format.
"""
import netrc, requests
from html.parser import HTMLParser

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, val in attrs:
                if attr == 'href' and '.tar' in val:
                    self.links.append(val)

def make_session():
    n = netrc.netrc()
    auth = n.authenticators("urs.earthdata.nasa.gov")
    session = requests.Session()
    session.auth = (auth[0], auth[2])
    session.rebuild_auth = lambda prepared, response: None
    return session

session = make_session()

# Search day 160 2021 directory for our target stations
url = "https://cddis.nasa.gov/archive/gnss/data/highrate/2021/160/"
r = session.get(url, allow_redirects=True, timeout=30)

parser = LinkParser()
parser.feed(r.text)

# Search for stations we want
targets = ['REYK', 'ONSA', 'LAMA', 'BOR1', 'POTS', 'WSRT', 'BRST', 'ISTA']
print(f"Found {len(parser.links)} tar files total\n")
print("Target stations found:")
for link in parser.links:
    for t in targets:
        if t in link.upper():
            print(f"  {link}")

print("\nAll available stations (first 30):")
for link in parser.links[:30]:
    print(f"  {link}")

"""This file contains code to scrape the fixture website with the scores of previous matches."""
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

class FetchFixtures:
    def get(self):
        response = requests.get(f"https://fixturedownload.com/sport/football")

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        pars = soup.find_all('a', href=True)

        for par in tqdm(pars):
            if 'href' in par.attrs:
                if 'json' in par.attrs['href']:
                    with open(f"fixtures/{par.attrs['href'].split('/')[-2]}_{par.attrs['href'].split('/')[-1]}.json",
                              'w') as file:
                        url = "https://fixturedownload.com" + par.attrs['href'].replace('view', 'feed')
                        url_response = requests.get(url)
                        file.write(url_response.content.decode("utf-8"))


if __name__ == '__main__':
    FetchFixtures().get()
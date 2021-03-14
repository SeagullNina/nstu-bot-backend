import requests
import csv
from bs4 import BeautifulSoup as BS
from GrabzIt import GrabzItTableOptions
from GrabzIt import GrabzItClient

def minBalls():
    r = requests.get("https://www.nstu.ru/entrance/committee/exam_min")
    soup = BS(r.content)
    headersfirst = []
    minballsfirst = []
    blockfirst = soup.find_all('div', class_='table-responsive')[0]
    tablefirst = blockfirst.find('table')
    for tr in tablefirst.find_all('td', width='362'):
        headersfirst.append(tr.text.strip())
    for tr in tablefirst.find_all('td', width='254'):
        minballsfirst.append(tr.text.strip())
    with open('minBallsMiddle.csv', 'w') as f:
        writer = csv.writer(f, delimiter=';')
        level_counter = 0
        max_levels = len(minballsfirst)
        while level_counter < max_levels:
            writer.writerow((headersfirst[level_counter], minballsfirst[level_counter]))
            level_counter = level_counter + 1

def plan():
    grabzIt = GrabzItClient.GrabzItClient("ZWE0YzA5YmU2MTFiNDFhNTkwNTRjMWI5M2I5NTM4MjY=",
                                          "TVo/IX0/Cz8/Pz8/PzIAbD8aOj9wPxMBPzJQVlg0bQg=")

    options = GrabzItTableOptions.GrabzItTableOptions()
    options.format = 'xlsx'
    options.includeAllTables = True

    grabzIt.URLToTable("https://www.nstu.ru/entrance/committee/plan", options)
    grabzIt.SaveTo("plan.xlsx")

minBalls()
plan()
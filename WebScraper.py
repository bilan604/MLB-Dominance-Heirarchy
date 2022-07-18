import re
import requests
from bs4 import BeautifulSoup


def getOutcome(s1, s2):
    if s1 == s2:
        return 0
    if s1 > s2:
        return 1
    else:
        return 2


link_raw = "https://www.baseball-reference.com/leagues/majors/2021-schedule.shtml"
link_data = requests.get(link_raw, headers={"Content-Type": "html"})
soup = BeautifulSoup(link_data.text, 'html.parser')
p = soup.find_all('p')


team_1 = []
score_1 = []
team_2 = []
score_2 = []

# 1,2,4,5
for i in range(15, 2686):
    game = str(p[i]).split("\n")
    if len(game) == 8:
        team1 = game[1].split("\">")[1][:-4]
        team_1.append(team1)
        team2 = game[4].split("\">")[1][:-4]
        team_2.append(team2)
        score1 = re.sub("[^0-9]","",game[2])
        score_1.append(score1)
        score2 = re.sub("[^0-9]","",game[5])
        score_2.append(score2)

score_1 = [int(s) for s in score_1]
score_2 = [int(s) for s in score_2]
winning_team = [getOutcome(s1, s2) for s1, s2 in zip(score_1, score_2)]


df = pd.DataFrame({"Team1": team_1, "Team2": team_2, "Score1": score_1, "Score2": score_2, "WinningTeam": winning_team})


team_names = list(unique(team_1 + team_2))
wins = [np.sum(A[:, j]) for j in range(p)]
losses = [np.sum(A[i]) for i in range(p)]
WL_ratio = [w/l for w,l in zip(wins, losses)]


dd_teams = {"Name": team_names, "Wins": wins, "Losses": losses, "WLRatio": WL_ratio}
df_teams = pd.DataFrame(dd_teams)



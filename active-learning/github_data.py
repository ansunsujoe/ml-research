import json
from github import Github

# Get Github access token
with open("secrets.json") as f:
    data = json.load(f)
    token = data.get("github_token")

with open("issues-1.txt", "w") as f:
    g = Github(token)
    for x in g.get_repos():
        print("Processing repo " + x.name)
        for y in x.get_issues():
            f.write(y.title.replace("\n", " ") + "\n")
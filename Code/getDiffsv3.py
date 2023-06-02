import math
import os
import requests
import time
import sys
import json
import datetime
import concurrent.futures

from tqdm import tqdm

# get access token
if not os.path.isfile('access_token'):
    print("please place a Github access token in this directory.")
    sys.exit()
with open('access_token', 'r') as accestoken:
    access = accestoken.readline().replace("\n", "")

# get mined commits
repositories = {}
with open('all_commits.json', 'r') as infile:
    repositories = json.load(infile)

datafilter = {}

# load progress
if os.path.isfile('DataFilter.json'):
    with open('DataFilter.json', 'r') as infile:
        datafilter = json.load(infile)

if not 'showcase' in datafilter:
    datafilter['showcase'] = {}
if not 'no-python' in datafilter:
    datafilter['no-python'] = {}
if not 'python' in datafilter:
    datafilter['python'] = {}

print(str(len(datafilter['showcase'])) + " repositories are showcases and therefore ignored.")
print(str(len(datafilter['no-python'])) + " repositories don't even contain ANY python.")
print(str(len(datafilter['python'])) + " might contain python.")

data = {}

myheaders = {'Authorization': 'token ' + access}
progress = 0
total = 0
newrepos = 0
nopythonlist = {}  # used to mark which commits have and don't have python files modified


def process_repo(repo):
    global total, saved
    name = repo.split('https://github.com/')[1]

    if (name in datafilter['showcase']):
        print("skip: showcase")
        return
    if (name in datafilter['no-python']):
        print("skip: no python")
        return

    # print("\n" + repo + "     " + str(progress))

    if not repo in nopythonlist:
        nopythonlist[repo] = {}

    noPythonAtAll = True

    for c in repositories[repo]:
        # go through all commits of that repository

        if c in nopythonlist[repo]:
            # if we already know that the commit has no python, skip it
            continue

        # otherwise, get the DIFF file
        target = repo + '/commit/' + c + '.diff'
        response = requests.get(target, headers=myheaders)
        # this is the diff
        content = response.content
        try:
            diffcontent = content.decode('utf-8', errors='ignore');
        except:
            print("an exception occured. Skip.");
            continue;

        # check if the file contains any python
        if (".py" in diffcontent):
            noPythonAtAll = False  # the repository in general contains at least some python

            # put it in data
            if not repo in data:
                data[repo] = {}

                # we should save again when the time is right
            total = total + 1
            saved = False

            # copy the relevant information to 'data'
            data[repo][c] = repositories[repo][c]
            data[repo][c]["diff"] = content.decode('utf-8', errors='ignore');

        else:
            if not repo in nopythonlist:
                nopythonlist[repo] = {}
            # note down that this commit doesn't contain any pyhon files
            nopythonlist[repo][c] = {}

    if noPythonAtAll:
        # note down that this repository doesn't contain any python files
        datafilter['no-python'][name] = {}
    else:
        # repository has some python, and we checked it now
        datafilter['python'][name] = {}



with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    saved = True
    for repo in repositories:
        futures.append(executor.submit(process_repo, repo))
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(repositories)):
        # handle the completed future, if needed
        pass

print(str(total) + " commits modifying python were found.")

# save the markings of what contains no python data
with open('DataFilter.json', 'w') as outfile:
    json.dump(datafilter, outfile)

#save the actual python results
with open('PyCommitsWithDiffsv2.json', 'w') as outfile:
    json.dump(data, outfile)



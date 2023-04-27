import requests
import time
import sys
import ujson as json
import os
import concurrent.futures


def load_json(file_path):
    data = {}
    if os.path.isfile(file_path):
        with open(file_path, 'r') as infile:
            data = json.load(infile)
    return data


def save_json(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)


toomuchsecurity = {'offensive', 'pentest', 'vulnerab', 'security', 'hack', 'exploit', 'ctf ', ' ctf',
                   'capture the flag'}
alittletoomuch = {'offensive security', 'pentest', 'exploits', 'vulnerability research', 'hacking',
                  'security framework', 'vulnerability database', 'simulated attack', 'security research'}

# load mined commits data
repositories = load_json('all_commits.json')

# load progress data
data = load_json('DataFilter.json')
if not "showcase" in data:
    data["showcase"] = {}
if not "noshowcase" in data:
    data["noshowcase"] = {}

# get access token
if not os.path.isfile('access_token'):
    print("please place a Github access token in this directory.")
    sys.exit()
with open('access_token', 'r') as accestoken:
    access = accestoken.readline().replace("\n", "")
myheaders = {'Authorization': 'token ' + access}


# define a function to get the showcase status of a repository
def get_showcase_status(name):
    # check if we already know the showcase status of this repository
    if (name in data['showcase']) or (name in data['noshowcase']):
        return

    # check if any forbidden keyword is present in the name
    if toomuchsecurity.intersection(name.split('/')):
        data['showcase'][name] = {}
        print(name + ": showcase")
        return

    # get the readme of the repository
    response = requests.get('https://github.com/' + name + '/blob/master/README.md', headers=myheaders)
    h = response.headers

    if ("markdown-body") in response.text:
        # find the description of the project
        pos = response.text.find("markdown-body")
        pos2 = response.text.find("/article")
        description = response.text[pos:pos2]

        # check if any forbidden keyword is present in the description
        if alittletoomuch.intersection(description.split()):
            data['showcase'][name] = {}
            print(name + ": showcase")
            return

    # put it in the "noshowcase" category
    print(name + ": not a showcase")
    data['noshowcase'][name] = {}


# create a thread pool executor with 10 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # submit each repository name to the executor
    futures = [executor.submit(get_showcase_status, repo.split('https://github.com/')[1]) for repo in repositories]
    # wait for all the futures to complete
    concurrent.futures.wait(futures)

# save progress data
save_json(data, 'DataFilter.json')
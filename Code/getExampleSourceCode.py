import random
import sys

import ujson as json

# default mode / type of vulnerability
mode = "sql"

# get the vulnerability from the command line argument
if (len(sys.argv) > 1):
    mode = sys.argv[1]

### paramters for the filtering and creation of samples
restriction = [20000, 5, 6, 10]  # which samples to filter out
step = 5  # step length n in the description
fulllength = 200  # context length m in the description
# load data
with open(f'data/plain_{mode}', 'r', encoding='utf-8') as infile:
    data = json.load(infile)


def get_source_codes(data, repositories):
    source_codes = []
    badparts = []
    goodparts = []
    msg = []
    for repository in data:
        for commit in data[repository]:
            msg.append(data[repository][commit]['msg'])
            files = data[repository][commit]['files']
            for file_key in files:
                file = files[file_key]
                if 'sourceWithComments' in file:
                    source_codes.append(file['sourceWithComments'])
                    for change in file.get("changes", []):
                        badparts = change.get("badparts", [])
                        goodparts = change.get("goodparts", [])
                        badparts.extend(badparts)
                        goodparts.extend(goodparts)

    return source_codes, badparts, goodparts, msg


def save_formatted_code(source_codes, mode):
    for i in range(len(source_codes)):
        file_name = f"examples/{mode}-{i + 1}.py"

        with open(file_name, "w") as file:
            file.write(source_codes[i])


amount = 10
repositories = list(data.keys()) if len(data) < amount else random.sample(list(data.keys()), amount)
source_codes, badparts, goodparts, msg = get_source_codes(data, repositories)
for i, j in zip(range(len(source_codes)), range(len(badparts))):
    print(f"badparts {j + 1}:")
    print(badparts[j])
    print(f"goodparts {j + 1}:")
    print(goodparts[j])
    print(f"msg {i + 1}:")
    print(msg[i])
    print()

exit(0)
save_formatted_code(source_codes, mode)

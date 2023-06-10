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


def get_source_codes(json_file, repositories):
    source_codes = []
    for repository in repositories:
        for commit in json_file[repository]:
            if "files" not in json_file[repository][commit]:
                continue
            for file in json_file[repository][commit]["files"]:
                if "source" not in json_file[repository][commit]["files"][file]:
                    continue
                source_code = json_file[repository][commit]["files"][file]["source"]
                source_codes.append(source_code)
    return source_codes


def save_formatted_code(source_codes, mode):
    for i in range(len(source_codes)):
        file_name = f"examples/{mode}-{i + 1}.py"

        with open(file_name, "w") as file:
            file.write(source_codes[i])


amount = 10
repositories = list(data.keys()) if len(data) < amount else random.sample(list(data.keys()), amount)
source_codes = get_source_codes(data, repositories)
save_formatted_code(source_codes, mode)
